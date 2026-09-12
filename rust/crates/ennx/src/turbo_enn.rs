use ennx::config::load_turbo_enn_config;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

mod tune_output;
use tune_output::Terminal;

fn main() -> ExitCode {
    let mut terminal = Terminal::new(anstream::stderr());
    let status = match run(&mut terminal) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            terminal.send(format!("ennx tune: {error}\n").as_bytes());
            ExitCode::FAILURE
        }
    };
    terminal.finish();
    status
}

fn run_command(
    mut command: Command,
    mut log: &std::fs::File,
    terminal: &mut Terminal,
) -> Result<(), String> {
    let (reader, writer) = std::io::pipe().map_err(|error| error.to_string())?;
    command
        .stdout(Stdio::from(
            writer.try_clone().map_err(|error| error.to_string())?,
        ))
        .stderr(Stdio::from(writer));
    let mut child = command.spawn().map_err(|error| error.to_string())?;
    drop(command);
    let streamed = (|| -> std::io::Result<()> {
        let mut reader = std::io::BufReader::new(reader);
        let mut line = Vec::new();
        while reader.read_until(b'\n', &mut line)? != 0 {
            log.write_all(&line)?;
            terminal.send(&line);
            line.clear();
        }
        log.flush()
    })();
    if streamed.is_err() {
        let _ = child.kill();
    }
    let status = child.wait().map_err(|error| error.to_string())?;
    streamed.map_err(|error| format!("stream worker output: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command failed ({status})"))
    }
}

struct RunArtifacts {
    output: PathBuf,
    resolved_config: PathBuf,
    log: std::fs::File,
}

impl RunArtifacts {
    fn create(output_root: PathBuf, resolved: &str) -> Result<Self, String> {
        std::fs::create_dir_all(&output_root).map_err(|error| error.to_string())?;
        let output = allocate_output_directory(&output_root)?;
        let resolved_config = output.join("study.toml");
        std::fs::write(&resolved_config, resolved).map_err(|error| error.to_string())?;
        write_source_snapshot(&output)?;
        let log =
            std::fs::File::create(output.join("run.log")).map_err(|error| error.to_string())?;
        Ok(Self {
            output,
            resolved_config,
            log,
        })
    }

    fn record_result(&self, result: &Result<(), String>, summary: &str) -> Result<(), String> {
        (&self.log)
            .write_all(summary.as_bytes())
            .map_err(|error| error.to_string())?;
        let status = if result.is_ok() {
            "success\n"
        } else {
            "failure\n"
        };
        std::fs::write(self.output.join("exit.txt"), status).map_err(|error| error.to_string())?;
        if let Err(error) = result {
            if !self.output.join("result.toml").exists() {
                let message = error.replace('"', "'");
                std::fs::write(
                    self.output.join("result.toml"),
                    format!("status = \"failure\"\nstage = \"runner\"\nmessage = {message:?}\n"),
                )
                .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    }
}

fn allocate_output_directory(root: &Path) -> Result<PathBuf, String> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis();
    (0..100)
        .find_map(|suffix| {
            let path = root.join(format!("run-{timestamp}-{}-{suffix}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(Ok(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error.to_string())),
            }
        })
        .ok_or_else(|| "could not allocate a unique artifact directory".to_string())?
}

fn write_source_snapshot(output: &Path) -> Result<(), String> {
    let commands = [
        Command::new("jj")
            .args(["log", "-r", "@", "--no-graph", "-T", "commit_id"])
            .output(),
        Command::new("jj")
            .args(["diff", "--summary", "--no-pager"])
            .output(),
    ];
    let mut source =
        std::fs::File::create(output.join("source.txt")).map_err(|error| error.to_string())?;
    for result in commands {
        match result {
            Ok(result) if result.status.success() => {
                source
                    .write_all(&result.stdout)
                    .map_err(|error| error.to_string())?;
                source.write_all(b"\n").map_err(|error| error.to_string())?;
            }
            _ => source
                .write_all(b"unavailable\n")
                .map_err(|error| error.to_string())?,
        }
    }
    Ok(())
}

fn worker_command(config: &Path, output: &Path) -> Command {
    let mut worker = Command::new("./buck2w");
    worker
        .args(["run", "//rust/crates/ennx:turbo-enn-worker", "--"])
        .arg(config)
        .arg(output);
    worker
}

fn run(terminal: &mut Terminal) -> Result<(), String> {
    let input = std::env::args_os()
        .nth(1)
        .ok_or("expected a TuRBO-ENN TOML path")?;
    let (run, resolved) = load_turbo_enn_config(Path::new(&input))?;
    let artifacts = RunArtifacts::create(run.output(), &resolved)?;
    let title = if run.study == Some(ennx::TurboEnnStudy::Pretrain) {
        "ENNX pretrain"
    } else {
        "ENNX diagnostics"
    };
    terminal.send(format!("{title}\n").as_bytes());
    let started = std::time::Instant::now();
    let result = run_command(
        worker_command(&artifacts.resolved_config, &artifacts.output),
        &artifacts.log,
        terminal,
    );
    let summary = format!(
        "\n{} in {:.1}s\nArtifacts: {}\n",
        if result.is_ok() {
            "Completed"
        } else {
            "Failed"
        },
        started.elapsed().as_secs_f64(),
        artifacts.output.display(),
    );
    artifacts.record_result(&result, &summary)?;
    if terminal.skipped > 0 {
        writeln!(
            &artifacts.log,
            "Display skipped {} messages; worker records above are complete",
            terminal.skipped
        )
        .map_err(|error| error.to_string())?;
    }
    terminal.send(summary.as_bytes());
    result.map_err(|error| {
        format!(
            "TuRBO-ENN round study failed: {error}; see {}/run.log",
            artifacts.output.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.log");
        let log = std::fs::File::create(&path).unwrap();
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "printf 'out\\n'; printf 'err\\n' >&2; printf 'tail'; exit 7",
        ]);
        let mut terminal = Terminal::new(std::io::sink());
        assert!(run_command(command, &log, &mut terminal)
            .unwrap_err()
            .contains("7"));
        terminal.finish();
        assert_eq!(std::fs::read_to_string(path).unwrap(), "out\nerr\ntail");
    }

    #[test]
    fn live_output() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.log");
        let ack = directory.path().join("ack");
        let log = std::fs::File::create(&path).unwrap();
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "printf 'ready\\n'; while [ ! -f \"$1\" ]; do sleep 0.01; done; printf 'done\\n'",
            "test",
        ]);
        command.arg(&ack);
        let worker = std::thread::spawn(move || {
            let mut terminal = Terminal::new(std::io::sink());
            let result = run_command(command, &log, &mut terminal);
            terminal.finish();
            result
        });
        let started = std::time::Instant::now();
        let mut seen = false;
        while started.elapsed().as_secs() < 5 {
            if std::fs::read_to_string(&path).unwrap() == "ready\n" {
                seen = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        std::fs::write(ack, "").unwrap();
        worker.join().unwrap().unwrap();
        assert!(seen, "output must reach the log before the worker exits");
        assert_eq!(std::fs::read_to_string(path).unwrap(), "ready\ndone\n");
    }

    #[test]
    fn round_display() {
        let line = b"TURBO_ENN_ACTUAL_ROUND round=2 total=100 wall_seconds=0.88 scorer_gpu_seconds=0.87 reward=-9 variance=0.01 accepted=false radius=0.01 proposal_radius=0.005 changed_weights=95 parameters=100\n";
        let rendered = tune_output::display_line(line);
        let mut plain = anstream::StripStream::new(Vec::new());
        plain.write_all(rendered.as_bytes()).unwrap();
        let plain = String::from_utf8(plain.into_inner()).unwrap();
        assert_eq!(
            plain,
            "   2/100     880ms    870ms   9.00000  95.00%   0.500%  rejected\n"
        );
        assert_eq!(
            tune_output::display_line(b"TURBO_ENN_TRUST round=10 outcome=Inconclusive\n"),
            ""
        );
        assert_eq!(tune_output::display_line(b"diagnostic\n"), "diagnostic\n");
        assert_eq!(
            tune_output::display_line(b"TURBO_ENN_ACTUAL_ROUND broken\n"),
            "TURBO_ENN_ACTUAL_ROUND broken\n"
        );
        assert_eq!(
            tune_output::display_line(b"[2026-09-28] File changed: x\n"),
            ""
        );
        assert_eq!(
            tune_output::display_line(b"[2026-09-28] error: link failed\n"),
            "[2026-09-28] error: link failed\n"
        );
    }

    #[test]
    fn slow_terminal() {
        struct Blocked(std::sync::mpsc::Receiver<()>);
        impl Write for Blocked {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let _ = self.0.recv();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.log");
        let log = std::fs::File::create(&path).unwrap();
        let (release, blocked) = std::sync::mpsc::channel();
        let (finished, done) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut terminal = Terminal::new(Blocked(blocked));
            let mut command = Command::new("sh");
            command.args([
                "-c",
                "i=0; while [ $i -lt 1000 ]; do printf 'record\\n'; i=$((i+1)); done",
            ]);
            let result = run_command(command, &log, &mut terminal);
            let skipped = terminal.skipped;
            terminal.finish();
            let _ = finished.send((result, skipped));
        });
        let outcome = done.recv_timeout(std::time::Duration::from_secs(5));
        drop(release);
        worker.join().unwrap();
        let (result, skipped) =
            outcome.expect("terminal must not block log collection or shutdown");
        result.unwrap();
        assert!(skipped > 0);
        assert_eq!(
            std::fs::read_to_string(path).unwrap(),
            "record\n".repeat(1000)
        );
    }

    #[test]
    fn closed_terminal() {
        struct Closed;
        impl Write for Closed {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.log");
        let log = std::fs::File::create(&path).unwrap();
        let mut terminal = Terminal::new(Closed);
        let mut command = Command::new("sh");
        command.args(["-c", "printf 'first\\nlast\\n'"]);
        run_command(command, &log, &mut terminal).unwrap();
        terminal.finish();
        assert_eq!(std::fs::read_to_string(path).unwrap(), "first\nlast\n");
    }
}
