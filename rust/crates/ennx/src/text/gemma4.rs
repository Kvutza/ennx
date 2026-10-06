//! Gemma 4 Prompt Formatting and Control Token Specification.
//!
//! Implements Google DeepMind's Gemma 4 Prompt Formatting standard:
//! - Turn delimiters: `<|turn>` and `<turn|>` with `system`, `user`, `model` roles.
//! - Thought channel: `<|channel>thought\n...\n<channel|>` with thought stripping.
//! - Empty thought channel representation stabilization: `<|channel>thought\n<channel|>`.
//! - Tool lifecycle: `<|tool>`, `<tool|>`, `<|tool_call>`, `<tool_call|>`, `<|tool_response>`, `<tool_response|>`.
//! - String delimiters: `<|"|>`.
//! - Learnability evaluation to detect and prevent cyclical repetition loops.

use std::collections::HashSet;

/// Gemma 4 special token integer IDs from DeepMind's Gemma 4 tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4SpecialToken {
    Pad = 0,
    Eos = 1,
    Bos = 2,
    Unk = 3,
    Mask = 4,
    BeginOfToolResponse = 50,
    StartOfTurn = 105,
    EndOfTurn = 106,
    StartOfImage = 255999,
    StartOfAudio = 256000,
    ImagePlaceholder = 258880,
    AudioPlaceholder = 258881,
    EndOfImage = 258882,
    EndOfAudio = 258883,
}

/// Gemma 4 control token string literals.
pub mod tags {
    pub const START_OF_TURN: &str = "<|turn>";
    pub const END_OF_TURN: &str = "<turn|>";
    pub const THINK: &str = "<|think|>";
    pub const START_OF_CHANNEL: &str = "<|channel>";
    pub const END_OF_CHANNEL: &str = "<channel|>";
    pub const START_OF_TOOL: &str = "<|tool>";
    pub const END_OF_TOOL: &str = "<tool|>";
    pub const START_OF_TOOL_CALL: &str = "<|tool_call>";
    pub const END_OF_TOOL_CALL: &str = "<tool_call|>";
    pub const START_OF_TOOL_RESPONSE: &str = "<|tool_response>";
    pub const END_OF_TOOL_RESPONSE: &str = "<tool_response|>";
    pub const STRING_DELIMITER: &str = "<|\"|>";
    pub const IMAGE_PLACEHOLDER: &str = "<|image|>";
    pub const AUDIO_PLACEHOLDER: &str = "<|audio|>";
}

/// Gemma 4 conversation roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4Role {
    System,
    User,
    Model,
}

impl Gemma4Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Model => "model",
        }
    }
}

/// Gemma 4 Prompt Formatter.
pub struct Gemma4Formatter;

impl Gemma4Formatter {
    /// Format a system turn with optional thinking trigger `<|think|>`.
    pub fn format_system(instructions: &str, think: bool) -> String {
        let mut turn = String::from(tags::START_OF_TURN);
        turn.push_str("system\n");
        if think {
            turn.push_str(tags::THINK);
        }
        turn.push_str(instructions);
        turn.push_str(tags::END_OF_TURN);
        turn
    }

    /// Format a user dialogue turn.
    pub fn format_user(content: &str) -> String {
        let mut turn = String::from(tags::START_OF_TURN);
        turn.push_str("user\n");
        turn.push_str(content);
        turn.push_str(tags::END_OF_TURN);
        turn
    }

    /// Format a model turn with an internal thought channel.
    pub fn format_model(thought: Option<&str>, content: &str) -> String {
        let mut turn = String::from(tags::START_OF_TURN);
        turn.push_str("model\n");
        if let Some(reasoning) = thought {
            turn.push_str(tags::START_OF_CHANNEL);
            turn.push_str("thought\n");
            turn.push_str(reasoning);
            turn.push('\n');
            turn.push_str(tags::END_OF_CHANNEL);
        }
        turn.push_str(content);
        turn.push_str(tags::END_OF_TURN);
        turn
    }

    /// Format an empty thought channel for representation stabilization in fine-tuning.
    pub fn format_stabilized_model(content: &str) -> String {
        let mut turn = String::from(tags::START_OF_TURN);
        turn.push_str("model\n");
        turn.push_str(tags::START_OF_CHANNEL);
        turn.push_str("thought\n");
        turn.push_str(tags::END_OF_CHANNEL);
        turn.push_str(content);
        turn.push_str(tags::END_OF_TURN);
        turn
    }

    /// Format a tool call with `<|"|>` delimiters.
    pub fn format_tool_call(function_name: &str, arguments: &[(&str, &str)]) -> String {
        let mut out = String::from(tags::START_OF_TOOL_CALL);
        out.push_str("call:");
        out.push_str(function_name);
        out.push('{');
        for (idx, (k, v)) in arguments.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            out.push_str(k);
            out.push(':');
            out.push_str(tags::STRING_DELIMITER);
            out.push_str(v);
            out.push_str(tags::STRING_DELIMITER);
        }
        out.push('}');
        out.push_str(tags::END_OF_TOOL_CALL);
        out
    }

    /// Format a tool response block.
    pub fn format_tool_response(function_name: &str, response_body: &str) -> String {
        let mut out = String::from(tags::START_OF_TOOL_RESPONSE);
        out.push_str("response:");
        out.push_str(function_name);
        out.push('{');
        out.push_str(response_body);
        out.push('}');
        out.push_str(tags::END_OF_TOOL_RESPONSE);
        out
    }

    /// Strip private thought channels between turns to prevent cyclical reasoning loops.
    pub fn strip_thoughts(dialogue: &str) -> String {
        let mut result = String::with_capacity(dialogue.len());
        let mut remaining = dialogue;
        while let Some(start_idx) = remaining.find(tags::START_OF_CHANNEL) {
            result.push_str(&remaining[..start_idx]);
            let channel_body = &remaining[start_idx + tags::START_OF_CHANNEL.len()..];
            if let Some(end_idx) = channel_body.find(tags::END_OF_CHANNEL) {
                remaining = &channel_body[end_idx + tags::END_OF_CHANNEL.len()..];
            } else {
                remaining = "";
                break;
            }
        }
        result.push_str(remaining);
        result
    }

    /// Extract private thought channel content if present.
    pub fn extract_thought(model_turn: &str) -> Option<String> {
        let start = model_turn.find(tags::START_OF_CHANNEL)?;
        let rest = &model_turn[start + tags::START_OF_CHANNEL.len()..];
        let rest = rest.strip_prefix("thought\n").unwrap_or(rest);
        let end = rest.find(tags::END_OF_CHANNEL)?;
        Some(rest[..end].trim().to_owned())
    }
}

/// Metrics measuring prompt learnability and absence of degenerate repetition attractors.
#[derive(Debug, Clone, PartialEq)]
pub struct LearnabilityReport {
    pub turns_count: usize,
    pub turns_balanced: bool,
    pub channel_balanced: bool,
    pub repetition_ratio_4gram: f32,
    pub thought_tokens_fraction: f32,
    pub learnable: bool,
}

/// Evaluate learnability and syntax structure of generated or prompt text.
pub fn evaluate_learnability(text: &str) -> LearnabilityReport {
    let start_turns = text.matches(tags::START_OF_TURN).count();
    let end_turns = text.matches(tags::END_OF_TURN).count();
    let turns_balanced = start_turns > 0 && start_turns == end_turns;

    let start_channels = text.matches(tags::START_OF_CHANNEL).count();
    let end_channels = text.matches(tags::END_OF_CHANNEL).count();
    let channel_balanced = start_channels == end_channels;

    let total_chars = text.len().max(1);
    let thought_chars: usize = text
        .match_indices(tags::START_OF_CHANNEL)
        .filter_map(|(start, _)| {
            let rest = &text[start..];
            let end = rest.find(tags::END_OF_CHANNEL)?;
            Some(end)
        })
        .sum();
    let thought_tokens_fraction = (thought_chars as f32 / total_chars as f32).min(1.0);

    let words: Vec<&str> = text.split_whitespace().collect();
    let repetition_ratio_4gram = if words.len() >= 4 {
        let mut ngrams = HashSet::new();
        let total_ngrams = words.len() - 3;
        for window in words.windows(4) {
            ngrams.insert(window);
        }
        ngrams.len() as f32 / total_ngrams as f32
    } else {
        1.0
    };

    let learnable = repetition_ratio_4gram > 0.4 && (turns_balanced || start_turns == 0);

    LearnabilityReport {
        turns_count: start_turns,
        turns_balanced,
        channel_balanced,
        repetition_ratio_4gram,
        thought_tokens_fraction,
        learnable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_turn_formatting() {
        let system = Gemma4Formatter::format_system("You are an assistant.", true);
        assert!(system.starts_with("<|turn>system\n<|think|>"));
        assert!(system.ends_with("<turn|>"));

        let user = Gemma4Formatter::format_user("Write quicksort in Rust.");
        assert_eq!(user, "<|turn>user\nWrite quicksort in Rust.<turn|>");

        let model = Gemma4Formatter::format_model(
            Some("Need to pick a pivot."),
            "fn quicksort() {}",
        );
        assert!(model.contains("<|channel>thought\nNeed to pick a pivot.\n<channel|>"));
        assert!(model.contains("fn quicksort() {}"));
    }

    #[test]
    fn gemma4_empty_channel_stabilization() {
        let model = Gemma4Formatter::format_stabilized_model("pub fn hello() {}");
        assert_eq!(
            model,
            "<|turn>model\n<|channel>thought\n<channel|>pub fn hello() {}<turn|>"
        );
    }

    #[test]
    fn gemma4_thought_stripping() {
        let turn = Gemma4Formatter::format_model(
            Some("Private chain of thought."),
            "Answer 42",
        );
        let stripped = Gemma4Formatter::strip_thoughts(&turn);
        assert!(!stripped.contains("Private chain of thought"));
        assert!(stripped.contains("Answer 42"));
        assert_eq!(
            Gemma4Formatter::extract_thought(&turn).as_deref(),
            Some("Private chain of thought.")
        );
    }

    #[test]
    fn gemma4_tool_delimiters() {
        let call = Gemma4Formatter::format_tool_call(
            "read_file",
            &[("path", "src/main.rs")],
        );
        assert_eq!(
            call,
            "<|tool_call>call:read_file{path:<|\"|>src/main.rs<|\"|>}<tool_call|>"
        );
    }

    #[test]
    fn learnability_repetition_detection() {
        let repetitive = "Preprocessor ".repeat(500);
        let rep = evaluate_learnability(&repetitive);
        assert!(!rep.learnable);
        assert!(rep.repetition_ratio_4gram < 0.1);

        let good = "<|turn>system\nHelper<turn|><|turn>user\nHi<turn|><|turn>model\n<|channel>thought\n<channel|>Hello world today<turn|>";
        let rep_good = evaluate_learnability(good);
        assert!(rep_good.learnable);
        assert!(rep_good.turns_balanced);
        assert!(rep_good.channel_balanced);
    }
}
