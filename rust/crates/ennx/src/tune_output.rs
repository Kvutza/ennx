use std::io::Write;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::Duration;

const HEADING: anstyle::Style = anstyle::Style::new().bold();
const GOOD: anstyle::Style =
    anstyle::Style::new().fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Green)));
const MUTED: anstyle::Style = anstyle::Style::new().dimmed();

pub(super) struct Terminal {
    sender: SyncSender<Vec<u8>>,
    done: Receiver<()>,
    pub skipped: u64,
}

impl Terminal {
    pub fn new(mut output: impl Write + Send + 'static) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(32);
        let (finished, done) = mpsc::channel();
        std::thread::spawn(move || {
            while let Ok(line) = receiver.recv() {
                if output.write_all(display_line(&line).as_bytes()).is_err() {
                    break;
                }
            }
            let _ = output.flush();
            let _ = finished.send(());
        });
        Self {
            sender,
            done,
            skipped: 0,
        }
    }

    pub fn send(&mut self, line: &[u8]) {
        // Only the display is bounded; the caller preserves full worker records.
        let mut message = line[..line.len().min(16 * 1024)].to_vec();
        if message.len() < line.len() {
            message.extend_from_slice(b"... [full output in run.log]\n");
        }
        if self.sender.try_send(message).is_err() {
            self.skipped += 1;
        }
    }

    pub fn finish(self) {
        drop(self.sender);
        // A stalled terminal must not prevent process exit after saving artifacts.
        let _ = self.done.recv_timeout(Duration::from_millis(50));
    }
}

pub(super) fn display_line(line: &[u8]) -> String {
    let text = String::from_utf8_lossy(line);
    if let Some(rendered) = display_record(&text) {
        return rendered;
    }
    if text.starts_with("TURBO_ENN_TRUST ") {
        return String::new();
    }
    let message = text
        .strip_prefix('[')
        .and_then(|s| s.split_once("] "))
        .filter(|(stamp, _)| stamp.starts_with("20"))
        .map_or(text.as_ref(), |(_, message)| message);
    if [
        "Build ID:",
        "File changed:",
        "Directory changed:",
        "Network:",
        "Cache hits:",
        "Commands:",
        "BUILD SUCCEEDED - starting your binary",
    ]
    .iter()
    .any(|prefix| message.starts_with(prefix))
        || message
            .trim_end()
            .ends_with("additional file change events")
        || text.starts_with("[weights] per-tensor records:")
    {
        return String::new();
    }
    if let Some(stage) = text.strip_prefix("[tune] ") {
        return format!("{MUTED}{}{MUTED:#}\n", stage.trim_end());
    }
    text.into_owned()
}

fn display_record(text: &str) -> Option<String> {
    let field = |name: &str| {
        text.split_whitespace()
            .find_map(|part| part.split_once('=').filter(|(key, _)| *key == name))
            .map(|(_, value)| value)
    };
    let number = |name| {
        field(name)?
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
    };
    if text.starts_with("TURBO_ENN_ACTUAL_ROUND ") {
        let (style, decision) = match field("accepted")? {
            "true" => (GOOD, "applied"),
            "false" => (MUTED, "rejected"),
            _ => return None,
        };
        let parameters = number("parameters")?;
        if parameters <= 0.0 {
            return None;
        }
        return Some(format!(
            "{:>4}/{:<4} {:>6.0}ms {:>6.0}ms {:>9.5} {:>6.2}% {:>7.3}%  {style}{decision}{style:#}\n",
            field("round")?,
            field("total")?,
            number("wall_seconds")? * 1000.0,
            number("scorer_gpu_seconds")? * 1000.0,
            -number("reward")?,
            number("changed_weights")? / parameters * 100.0,
            number("proposal_radius")? * 100.0,
        ));
    }
    if text.starts_with("TURBO_ENN_FULL_SPACE ") {
        let proposals = match field("noise")? {
            "independent_rademacher" => "Rademacher",
            "independent_gaussian" => "Gaussian",
            _ => return None,
        };
        return Some(format!(
            "\n{HEADING}Model  {:.3}B weights | FP16{HEADING:#}\nContext {} | Batch {} | Full-weight {proposals} proposals\n",
            number("parameters")? / 1e9,
            field("context")?,
            field("batch")?,
        ));
    }
    if text.starts_with("TURBO_ENN_ACTUAL_INITIAL ") {
        return Some(format!(
            "Initial NLL  {:.5}\n\n{HEADING}Round        Wall   Scorer       NLL  Changed  Proposal  Decision{HEADING:#}\n",
            -number("reward")?,
        ));
    }
    if text.starts_with("TURBO_ENN_LOOP ") {
        return Some(format!(
            "\nLoop  {:.3}s | {:.3} rounds/s | includes round reporting\n",
            number("elapsed_seconds")?,
            number("rounds_per_second")?,
        ));
    }
    if text.starts_with("TURBO_ENN_SUMMARY ") {
        let met = match field("target_met")? {
            "true" => "met",
            "false" => "not met",
            _ => return None,
        };
        return Some(format!(
            "Round median {:.0}ms | Maximum {:.0}ms | Applied {}/{}\nTarget {:.0}ms: {met}\n",
            number("median_seconds")? * 1000.0,
            number("max_seconds")? * 1000.0,
            field("accepted")?,
            field("rounds")?,
            number("target_ms")?,
        ));
    }
    None
}
