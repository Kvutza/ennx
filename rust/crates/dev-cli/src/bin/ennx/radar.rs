use clap::Subcommand;
use std::path::Path;
use std::process::Command;

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub enum RadarAction {
    /// Search for code patterns with enclosing AST symbols and complexity.
    Find {
        /// Pattern or literal text to search.
        pattern: String,
        /// Treat pattern as regular expression instead of literal string.
        #[arg(long, short = 'e')]
        regex: bool,
        /// Source context lines around matches.
        #[arg(long, short = 'C', default_value_t = 0)]
        context: usize,
        /// Forward directly to raw ripgrep (rg) for line-level text search.
        #[arg(long)]
        plain: bool,
    },
    /// Transitive blast radius and reachability of a symbol across the call graph.
    Impact {
        /// Symbol identifier (e.g. "path/to/file.rs:fn_name" or "fn_name").
        symbol: String,
    },
    /// Direct and transitive callers of a symbol.
    Callers {
        /// Symbol identifier.
        symbol: String,
    },
    /// Direct and transitive callees called by a symbol.
    Callees {
        /// Symbol identifier.
        symbol: String,
    },
    /// Zero-regression quality delta against git HEAD baseline.
    Gate {
        /// Specific commit revision or comparison range (defaults to working copy vs HEAD).
        rev: Option<String>,
    },
    /// Token-budgeted context bundle for AI agents with Mermaid flowchart.
    Task {
        /// Task description in natural language.
        query: String,
        /// Include Mermaid architecture flowchart diagram.
        #[arg(long, default_value_t = true)]
        graph: bool,
    },
    /// Start the Model Context Protocol (MCP) server.
    Mcp,
}

pub fn run(root: &Path, action: RadarAction) -> Result<(), String> {
    match action {
        RadarAction::Find {
            pattern,
            regex,
            context,
            plain,
        } => run_find(root, &pattern, regex, context, plain),
        RadarAction::Impact { symbol } => run_arg(root, "--impact", &symbol),
        RadarAction::Callers { symbol } => run_arg(root, "--callers", &symbol),
        RadarAction::Callees { symbol } => run_arg(root, "--callees", &symbol),
        RadarAction::Gate { rev } => run_gate(root, rev.as_deref()),
        RadarAction::Task { query, graph } => run_task(root, &query, graph),
        RadarAction::Mcp => run_tool(root, &["--mcp"]),
    }
}

fn run_find(
    root: &Path,
    pattern: &str,
    regex: bool,
    context: usize,
    plain: bool,
) -> Result<(), String> {
    if plain {
        let mut command = Command::new("rg");
        command.current_dir(root);
        if context > 0 {
            command.arg(format!("-C{context}"));
        }
        command.arg(pattern);
        return super::execute(command);
    }
    let flag = if regex { "--regex" } else { "--grep" };
    let grep_arg = format!("{flag}={pattern}");
    let mut args = vec![".", &grep_arg];
    let ctx_arg = format!("--grep-context={context}");
    if context > 0 {
        args.push(&ctx_arg);
    }
    run_tool(root, &args)
}

fn run_arg(root: &Path, flag: &str, value: &str) -> Result<(), String> {
    let arg = format!("{flag}={value}");
    run_tool(root, &[".", &arg])
}

fn run_gate(root: &Path, rev: Option<&str>) -> Result<(), String> {
    if let Some(r) = rev {
        let arg = format!("--quality-delta={r}");
        run_tool(root, &[".", &arg])
    } else {
        run_tool(root, &[".", "--quality-delta"])
    }
}

fn run_task(root: &Path, query: &str, graph: bool) -> Result<(), String> {
    let task_arg = format!("--pack-task={query}");
    let mut args = vec![".", &task_arg];
    if graph {
        args.push("--with-graph");
    }
    run_tool(root, &args)
}

fn run_tool(root: &Path, args: &[&str]) -> Result<(), String> {
    super::dev_exec(root, "radar", args)
}
