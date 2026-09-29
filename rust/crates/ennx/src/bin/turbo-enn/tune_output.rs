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
        let required = line.starts_with(b"ENNX_GENERATED_TEXT");
        let failed = if required {
            self.sender.send(message).is_err()
        } else {
            self.sender.try_send(message).is_err()
        };
        if failed {
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
    if let Some(chunk) = text.strip_prefix("ENNX_GENERATED_TEXT_CHUNK ") {
        return chunk.to_owned();
    }
    if text.starts_with("ENNX_GENERATED_TEXT_END") {
        return "\n".to_owned();
    }
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

fn record_field<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.split_whitespace()
        .find_map(|part| part.split_once('=').filter(|(key, _)| *key == name))
        .map(|(_, value)| value)
}

fn generated_record(text: &str) -> Option<String> {
    let field = |name| record_field(text, name);
    let scope = if field("phase") == Some("initial") {
        "Initial".to_owned()
    } else {
        format!("Round {} candidate", field("round")?)
    };
    let decision = match field("accepted") {
        Some("true") => " | applied",
        Some("false") => " | rejected",
        _ => "",
    };
    let (label, value) = field("reward")
        .map(|reward| ("Reward", reward))
        .or_else(|| field("nll").map(|nll| ("NLL", nll)))?;
    Some(format!(
        "\n{HEADING}{scope} generated text{HEADING:#} | {} tokens | {label} {value}{decision}\n",
        field("tokens")?,
    ))
}

fn display_record(text: &str) -> Option<String> {
    if text.starts_with("ENNX_GENERATED_TEXT ") {
        return generated_record(text);
    }
    if text.starts_with("TURBO_ENN_ACTUAL_ROUND ") {
        return round_record(text);
    }
    if text.starts_with("TURBO_ENN_FULL_SPACE ") {
        return space_record(text);
    }
    if text.starts_with("TURBO_ENN_ACTUAL_INITIAL ") {
        return initial_record(text);
    }
    if text.starts_with("TURBO_ENN_LOOP ") {
        return loop_record(text);
    }
    match text.strip_prefix("TURBO_ENN_SUMMARY ") {
        Some(summary) => summary_record(summary),
        None => None,
    }
}

fn finite_field(text: &str, name: &str) -> Option<f64> {
    record_field(text, name)?
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn round_record(text: &str) -> Option<String> {
    let (style, decision) = match record_field(text, "accepted")? {
        "true" => (GOOD, "applied"),
        "false" => (MUTED, "rejected"),
        _ => return None,
    };
    let parameters = finite_field(text, "parameters")?;
    if parameters <= 0.0 {
        return None;
    }
    Some(format!(
        "{:>4}/{:<4} {:>6.0}ms {:>6.0}ms {:>9.5} {:>6.2}% {:>7.3}%  {style}{decision}{style:#}\n",
        record_field(text, "round")?,
        record_field(text, "total")?,
        finite_field(text, "wall_seconds")? * 1000.0,
        finite_field(text, "scorer_gpu_seconds")? * 1000.0,
        -finite_field(text, "reward")?,
        finite_field(text, "changed_weights")? / parameters * 100.0,
        finite_field(text, "proposal_radius")? * 100.0,
    ))
}

fn space_record(text: &str) -> Option<String> {
    let proposals = match record_field(text, "noise")? {
        "independent_rademacher" => "Rademacher",
        "independent_gaussian" | "independent_gaussian_ziggurat256_v1" => "Gaussian",
        _ => return None,
    };
    Some(format!(
        "\n{HEADING}Model  {:.3}B weights | FP16{HEADING:#}\nContext {} | Batch {} | Full-weight {proposals} proposals\n",
        finite_field(text, "parameters")? / 1e9,
        record_field(text, "context")?,
        record_field(text, "batch")?,
    ))
}

fn initial_record(text: &str) -> Option<String> {
    Some(format!(
        "Initial NLL  {:.5}\n\n{HEADING}Round        Wall   Scorer       NLL  Changed  Proposal  Decision{HEADING:#}\n",
        -finite_field(text, "reward")?,
    ))
}

fn loop_record(text: &str) -> Option<String> {
    Some(format!(
        "\nLoop  {:.3}s | {:.3} rounds/s | includes round reporting\n",
        finite_field(text, "elapsed_seconds")?,
        finite_field(text, "rounds_per_second")?,
    ))
}

fn summary_record(text: &str) -> Option<String> {
    let field = |name| record_field(text, name);
    let number = |name| {
        field(name)?
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
    };
    let met = match field("target_met")? {
        "true" => "met",
        "false" => "not met",
        _ => return None,
    };
    Some(format!(
        "Round median {:.0}ms | Maximum {:.0}ms | Applied {}/{}\nTarget {:.0}ms: {met}\n",
        number("median_seconds")? * 1000.0,
        number("max_seconds")? * 1000.0,
        field("accepted")?,
        field("rounds")?,
        number("target_ms")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::display_record;

    #[test]
    fn native_label() {
        let record =
            display_record("ENNX_GENERATED_TEXT round=1 tokens=4096 reward=0.12 accepted=false")
                .unwrap();
        assert!(record.contains("Reward 0.12"));
        assert!(record.contains("rejected"));
        assert!(!record.contains("NLL"));
    }

    #[test]
    fn legacy_records() {
        let record =
            display_record("ENNX_GENERATED_TEXT phase=initial tokens=4096 nll=3.1").unwrap();
        assert!(record.contains("NLL 3.1"));
    }
}
