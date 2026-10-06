//! Live telemetry dashboard and table formatting inspired by Ratatui.
//! Displays formatted round records, status pill badges, Gemma 4 metrics,
//! and rolling Braille sparklines.

use super::tui_canvas::inline_sparkline;

const GOOD: anstyle::Style = anstyle::Style::new()
    .bold()
    .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Green)));
const WARN: anstyle::Style = anstyle::Style::new()
    .bold()
    .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Yellow)));
const MUTED: anstyle::Style = anstyle::Style::new().dimmed();
const BOLD: anstyle::Style = anstyle::Style::new().bold();
const CYAN: anstyle::Style =
    anstyle::Style::new().fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Cyan)));

pub(super) fn table_header() -> &'static str {
    concat!(
        "┌──────┬──────────┬──────────┬───────────┬──────────┬──────────────┬────────────────┐\n",
        "│  Rd  │   Wall   │  Tokens  │  Reward   │ Changed  │ Rolling Grad │ Status Badge   │\n",
        "├──────┼──────────┼──────────┼───────────┼──────────┼──────────────┼────────────────┤\n"
    )
}

pub(super) fn generation_row(
    round: u64,
    tokens: u64,
    elapsed_ms: u64,
    reward: f64,
    changed_pct: f64,
    status: &str,
    history: &[f64],
) -> String {
    let (badge_style, badge_text) = if status == "accepted" || status == "applied" {
        (GOOD, "✔ ACCEPTED")
    } else {
        (MUTED, "✖ REJECTED")
    };
    let reward_style = if reward >= 0.0 { GOOD } else { WARN };
    let spark = inline_sparkline(history, 8);
    format!(
        "│ {:>4} │ {:>6}ms │ {:>8} │ {reward_style}{:>+9.4}{reward_style:#} │ {:>7.2}% │   {CYAN}{spark}{CYAN:#}   │ {badge_style}{badge_text:<14}{badge_style:#} │\n",
        round, elapsed_ms, tokens, reward, changed_pct,
    )
}

pub(super) fn gemma_record(round: u64, score: f64, rep_ratio: f64, status_badge: &str) -> String {
    let score_style = if score >= 0.9 { GOOD } else { CYAN };
    let rep_style = if rep_ratio < 0.05 { GOOD } else { WARN };
    let badge_style = if status_badge == "PASS" { GOOD } else { WARN };
    format!(
        "  {BOLD}↳ [Gemma 4]{BOLD:#} Rd {round} | Learnability: {score_style}{score:.4}{score_style:#} | 4-Gram Rep: {rep_style}{:.2}%{rep_style:#} | Gate: {badge_style}[{status_badge}]{badge_style:#}\n",
        rep_ratio * 100.0,
    )
}

pub(super) fn text_header(round: &str, tokens: &str, reward: &str, decision: &str) -> String {
    let cyan = CYAN;
    let bold = BOLD;
    format!(
        "\n{cyan}╭─ {bold}Round {round} Candidate Rollout ({tokens} tokens){bold:#} {cyan}─────────────────────────────╮{cyan:#}\n"
    )
}

pub(super) fn text_footer(reward: &str, decision: &str) -> String {
    let cyan = CYAN;
    let good = GOOD;
    let dec_badge = if decision.contains("applied") {
        format!("{good}[APPLIED]{good:#}")
    } else {
        format!("{MUTED}[REJECTED]{MUTED:#}")
    };
    format!(
        "{cyan}╰──────────────────────────────────────── Reward {reward} • {dec_badge} {cyan}─╯{cyan:#}\n\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header() {
        assert!(table_header().contains("┌──────┬"));
        assert!(table_header().contains("Status Badge"));
    }

    #[test]
    fn test_row() {
        let history = [0.01, 0.02, 0.03, 0.04];
        let row = generation_row(1, 1048576, 1850, 0.0312, 100.0, "accepted", &history);
        assert!(row.contains("1850ms"));
        assert!(row.contains("ACCEPTED"));
    }

    #[test]
    fn test_record() {
        let rec = gemma_record(1, 0.985, 0.0, "PASS");
        assert!(rec.contains("Gemma 4"));
        assert!(rec.contains("PASS"));
    }
}
