//! Provider-neutral coding-agent transcripts.

use deser::{Deserialize, Serialize};
use ennx_wire::json::Value;
use std::collections::BTreeSet;

pub const AGENT_SCHEMA: &str = "ennx.agent.v1";
const TOOLS_V1: [&str; 4] = ["read", "write", "edit", "bash"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct AgentTranscript {
    pub schema: String,
    pub episode_id: String,
    pub provenance: TaskProvenance,
    pub messages: Vec<AgentMessage>,
    pub termination: EpisodeTermination,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct TaskProvenance {
    pub dataset: String,
    pub task_id: String,
    pub source_revision: String,
    pub split: DataSplit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum DataSplit {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum EpisodeTermination {
    Completed,
    Cancelled,
    Truncated,
    ContextOverflow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(tag = "role", rename_all = "snake_case", deny_unknown_fields)]
pub enum AgentMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        segments: Vec<AssistantSegment>,
    },
    Tool {
        call_id: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AssistantSegment {
    Text {
        content: String,
    },
    ToolCall {
        call_id: String,
        name: String,
        arguments: Value,
    },
}

impl AgentTranscript {
    pub fn validate(&self) -> Result<(), String> {
        self.validate_header()?;
        let mut calls = BTreeSet::new();
        let mut pending = BTreeSet::new();
        let mut started = false;
        let mut has_user = false;
        let mut has_assistant = false;

        for message in &self.messages {
            match message {
                AgentMessage::System { content } => {
                    if started || content.is_empty() {
                        return Err(
                            "system messages must be nonempty and precede the conversation".into(),
                        );
                    }
                }
                AgentMessage::User { content } => {
                    started = true;
                    has_user = true;
                    if content.is_empty() || !pending.is_empty() {
                        return Err("user messages must be nonempty and cannot interrupt pending tool calls".into());
                    }
                }
                AgentMessage::Assistant { segments } => {
                    started = true;
                    has_assistant = true;
                    if segments.is_empty() || !pending.is_empty() {
                        return Err("assistant messages require content and cannot precede pending tool results".into());
                    }
                    for segment in segments {
                        self.validate_segment(segment, &mut calls, &mut pending)?;
                    }
                }
                AgentMessage::Tool { call_id, .. } => {
                    started = true;
                    if !pending.remove(call_id) {
                        return Err(format!("tool result {call_id:?} has no unresolved call"));
                    }
                }
            }
        }

        if !has_user || !has_assistant {
            return Err("transcript requires user and assistant messages".into());
        }
        if self.termination == EpisodeTermination::Completed && !pending.is_empty() {
            return Err("completed transcript has unresolved tool calls".into());
        }
        Ok(())
    }

    fn validate_header(&self) -> Result<(), String> {
        if self.schema != AGENT_SCHEMA {
            return Err(format!("unsupported agent schema {:?}", self.schema));
        }
        if [
            self.episode_id.as_str(),
            self.provenance.dataset.as_str(),
            self.provenance.task_id.as_str(),
            self.provenance.source_revision.as_str(),
        ]
        .into_iter()
        .any(str::is_empty)
            || self.messages.is_empty()
        {
            return Err("agent identity, provenance, and messages must be nonempty".into());
        }
        Ok(())
    }

    fn validate_segment(
        &self,
        segment: &AssistantSegment,
        calls: &mut BTreeSet<String>,
        pending: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        match segment {
            AssistantSegment::Text { content } => {
                if content.is_empty() {
                    return Err("assistant text segments must be nonempty".into());
                }
            }
            AssistantSegment::ToolCall {
                call_id,
                name,
                arguments,
            } => {
                if call_id.is_empty() || !calls.insert(call_id.clone()) {
                    return Err(format!(
                        "tool call identifier {call_id:?} is empty or duplicated"
                    ));
                }
                if !TOOLS_V1.contains(&name.as_str()) {
                    return Err(format!("tool {name:?} is not in {AGENT_SCHEMA}"));
                }
                if !arguments.is_map() {
                    return Err(format!("tool call {call_id:?} arguments must be an object"));
                }
                pending.insert(call_id.clone());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> AgentTranscript {
        ennx_wire::json::from_str(input).unwrap()
    }

    #[test]
    fn valid_episode() {
        let transcript = parse(
            r#"{
                "schema":"ennx.agent.v1",
                "episode_id":"repair-1",
                "provenance":{
                    "dataset":"heldout-repairs",
                    "task_id":"task-1",
                    "source_revision":"0123456789abcdef",
                    "split":"test"
                },
                "messages":[
                    {"role":"system","content":"Repair the repository."},
                    {"role":"user","content":"Fix the failing parser."},
                    {"role":"assistant","segments":[
                        {"type":"tool_call","call_id":"call-1","name":"read","arguments":{"path":"src/parser.rs"}}
                    ]},
                    {"role":"tool","call_id":"call-1","content":"fn parse() {}","is_error":false},
                    {"role":"assistant","segments":[
                        {"type":"text","content":"I found the parser."},
                        {"type":"tool_call","call_id":"call-2","name":"edit","arguments":{"path":"src/parser.rs","old":"{}","new":"{ Ok(()) }"}}
                    ]},
                    {"role":"tool","call_id":"call-2","content":"updated","is_error":false},
                    {"role":"assistant","segments":[{"type":"text","content":"Fixed."}]}
                ],
                "termination":"completed"
            }"#,
        );
        transcript.validate().unwrap();
    }

    #[test]
    fn unresolved_call() {
        let transcript = parse(
            r#"{
                "schema":"ennx.agent.v1",
                "episode_id":"repair-2",
                "provenance":{
                    "dataset":"heldout-repairs",
                    "task_id":"task-2",
                    "source_revision":"0123456789abcdef",
                    "split":"validation"
                },
                "messages":[
                    {"role":"user","content":"Read the parser."},
                    {"role":"assistant","segments":[
                        {"type":"tool_call","call_id":"call-1","name":"read","arguments":{"path":"src/parser.rs"}}
                    ]}
                ],
                "termination":"completed"
            }"#,
        );
        assert_eq!(
            transcript.validate().unwrap_err(),
            "completed transcript has unresolved tool calls"
        );
    }

    #[test]
    fn invalid_tools() {
        for (name, arguments, error) in [
            ("read", r#""src/parser.rs""#, "arguments must be an object"),
            ("browser", "{}", "is not in ennx.agent.v1"),
        ] {
            let input = format!(
                r#"{{
                    "schema":"ennx.agent.v1",
                    "episode_id":"repair-3",
                    "provenance":{{
                        "dataset":"heldout-repairs",
                        "task_id":"task-3",
                        "source_revision":"0123456789abcdef",
                        "split":"train"
                    }},
                    "messages":[
                        {{"role":"user","content":"Inspect the parser."}},
                        {{"role":"assistant","segments":[
                            {{"type":"tool_call","call_id":"call-1","name":"{name}","arguments":{arguments}}}
                        ]}}
                    ],
                    "termination":"cancelled"
                }}"#,
            );
            assert!(parse(&input).validate().unwrap_err().contains(error));
        }
    }

    #[test]
    fn cancelled_stream() {
        let transcript = parse(
            r#"{
                "schema":"ennx.agent.v1",
                "episode_id":"repair-4",
                "provenance":{
                    "dataset":"heldout-repairs",
                    "task_id":"task-4",
                    "source_revision":"0123456789abcdef",
                    "split":"validation"
                },
                "messages":[
                    {"role":"user","content":"Inspect the parser."},
                    {"role":"assistant","segments":[
                        {"type":"tool_call","call_id":"call-1","name":"read","arguments":{"path":"src/parser.rs"}}
                    ]}
                ],
                "termination":"cancelled"
            }"#,
        );
        transcript.validate().unwrap();
    }
}
