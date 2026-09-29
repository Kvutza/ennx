//! Fixed executable training tasks, not a similarity surrogate for correctness.
//! MBPP: https://arxiv.org/abs/2108.07732
//! Stronger tests: https://arxiv.org/abs/2305.01210 (EvalPlus).
//! Our added cases are ENNX contract checks, not an official EvalPlus benchmark.
use super::decode::Rollout;
use super::gen_protocol::{RewardResult, native_vector};
use crate::config::GenerationTask;
use crate::text::ByteDecoder;
use deser::{Deserialize, Serialize};
use ennx_wire::json::{Value, json};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
pub(super) struct Environment {
    schema: String,
    source: String,
    task_id: u32,
    split: String,
    pub(super) tokenizer: PathBuf,
    prompt: String,
    prompt_tokens: Vec<u32>,
    entrypoint: String,
    reference: String,
    negative: String,
    cases: Vec<Case>,
}

#[derive(Deserialize, Serialize)]
#[deser(deny_unknown_fields)]
struct Case {
    args: Vec<Value>,
    expected: Value,
}

impl Environment {
    pub(super) fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut spec: Self = ennx_wire::toml::from_str(&text).map_err(|e| e.to_string())?;
        if spec.schema != "ennx.code_environment.v1"
            || spec.source.trim().is_empty()
            || spec.split != "train"
            || spec.prompt.is_empty()
            || spec.reference.is_empty()
            || spec.negative.is_empty()
            || spec.entrypoint.is_empty()
            || !(2..=256).contains(&spec.cases.len())
        {
            return Err(
                "execution requires a versioned training task, provenance and 2..256 checks".into(),
            );
        }
        spec.tokenizer = path
            .parent()
            .ok_or("environment has no parent")?
            .join(&spec.tokenizer)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        Ok(spec)
    }

    pub(super) fn task(&self, tokenizer: &ByteDecoder) -> Result<GenerationTask, String> {
        if tokenizer.decode_bytes(&self.prompt_tokens)? != self.prompt.as_bytes() {
            return Err(
                "environment prompt tokens do not decode exactly to its source prefix".into(),
            );
        }
        Ok(GenerationTask {
            prompt: self.prompt_tokens.clone(),
            expected: Vec::new(),
            decoys: Vec::new(),
        })
    }

    pub(super) fn provenance(&self) -> Value {
        json!({"kind":"executable_code_training", "source":self.source,
            "task_id":self.task_id,"split":self.split,"checks":self.cases.len(),
            "reference_supplied_to_model":false,"teacher_forcing":false,
            "text_policy":"entire_unmodified_completion","benchmark_claim":false})
    }
}

pub(super) struct Executor {
    spec: Environment,
    interpreter: PathBuf,
    profile: String,
    timeout: Duration,
}

impl Executor {
    pub(super) fn new(
        environment: &Path,
        interpreter: &Path,
        timeout_ms: u64,
    ) -> Result<Self, String> {
        let spec = Environment::load(environment)?;
        if resident_bytes(std::process::id()).is_none() {
            return Err("resident-memory supervision is unavailable".into());
        }
        let interpreter = interpreter.canonicalize().map_err(|e| e.to_string())?;
        let profile = sandbox_profile(&interpreter)?;
        let executor = Self {
            spec,
            interpreter,
            profile,
            timeout: Duration::from_millis(timeout_ms),
        };
        // A disabled/broken OS sandbox is an infrastructure error, never a zero reward.
        let report = executor.run("", false)?;
        if report["syntax"] != true || report["interface"] != true {
            return Err("isolated Python compiler did not validate the task prefix".into());
        }
        Ok(executor)
    }

    fn run(&self, completion: &str, execute: bool) -> Result<Value, String> {
        let request = json!({"source":format!("{}{completion}",self.spec.prompt),
            "entrypoint":self.spec.entrypoint,
            "inputs":self.spec.cases.iter().map(|case| &case.args).collect::<Vec<_>>(),
            "execute":execute});
        let mut child = Command::new("/usr/bin/sandbox-exec")
            .args(["-p", &self.profile])
            .arg(&self.interpreter)
            .args(["-I", "-S", "-B", "-c", HARNESS])
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("isolated Python launch: {e}"))?;
        let result = (|| {
            let input = child.stdin.take().ok_or("missing evaluator input")?;
            write_request(input, &request)?;
            self.wait_child(&mut child)
        })();
        match result {
            Ok(None) => {}
            stopped => {
                let _ = child.kill();
                let _ = child.wait();
                return stopped.map(|reason| failed_report(reason.unwrap(), self.spec.cases.len()));
            }
        }
        let output = child.wait_with_output().map_err(|e| e.to_string())?;
        if !output.status.success() {
            if execute {
                let mut report = failed_report("execution_failed", self.spec.cases.len());
                report["exit_status"] = json!(output.status.to_string());
                return Ok(report);
            }
            return Err(format!(
                "isolated Python failed ({}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let mut report = ennx_wire::json::from_slice(&output.stdout)
            .map_err(|e| format!("invalid execution report: {e}"))?;
        self.judge(&mut report)?;
        Ok(report)
    }

    fn wait_child(&self, child: &mut Child) -> Result<Option<&'static str>, String> {
        let started = Instant::now();
        loop {
            if child.try_wait().map_err(|e| e.to_string())?.is_some() {
                return Ok(None);
            }
            if started.elapsed() >= self.timeout {
                return Ok(Some("timeout"));
            }
            match resident_bytes(child.id()) {
                Some(bytes) if bytes > 512 * 1024 * 1024 => return Ok(Some("memory_limit")),
                None if started.elapsed() > Duration::from_millis(100)
                    && child.try_wait().map_err(|e| e.to_string())?.is_none() =>
                {
                    return Err("candidate memory supervision failed".into());
                }
                _ => {}
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn judge(&self, report: &mut Value) -> Result<(), String> {
        let rows = report["cases"].as_seq().ok_or("missing execution cases")?;
        if rows.len() > self.spec.cases.len() {
            return Err("too many execution cases".into());
        }
        let mut rows = rows.to_vec();
        let mut passed = 0;
        for (index, (row, case)) in rows.iter_mut().zip(&self.spec.cases).enumerate() {
            let correct = report["syntax"] == true
                && report["interface"] == true
                && row["index"].as_u64() == Some(index as u64)
                && row.get("error").is_none()
                && row.get("actual") == Some(&case.expected);
            row["passed"] = json!(correct);
            passed += usize::from(correct);
        }
        report["cases"] = json!(rows);
        report["passed"] = json!(passed);
        Ok(())
    }

    pub(super) fn audit(&self) -> Result<Value, String> {
        let reference = self.run(&self.spec.reference, true)?;
        let negative = self.run(&self.spec.negative, true)?;
        let repetition = self.run("noise noise noise\n".repeat(128).as_str(), true)?;
        let passed = reference["passed"].as_u64() == Some(self.spec.cases.len() as u64)
            && negative["passed"].as_u64() == Some(0)
            && repetition["passed"].as_u64() == Some(0);
        Ok(
            json!({"passed":passed,"reference":reference,"negative":negative,
            "repetition":repetition,"provenance":self.spec.provenance(),
            "learning_quality_established":false}),
        )
    }

    pub(super) fn score(
        &self,
        tokenizer: &ByteDecoder,
        rollouts: &[Rollout],
        path: &Path,
    ) -> Result<RewardResult, String> {
        if rollouts.len() != 1 {
            return Err("execution requires one fixed training task".into());
        }
        let start = Instant::now();
        let bytes = tokenizer.decode_bytes(&rollouts[0].tokens)?;
        let mut report = match std::str::from_utf8(&bytes) {
            Ok(text) => self.run(text, true)?,
            Err(_) => failed_report("invalid_utf8", self.spec.cases.len()),
        };
        let passed = report["passed"]
            .as_u64()
            .ok_or("missing execution pass count")?;
        if passed > self.spec.cases.len() as u64 {
            return Err("invalid execution pass count".into());
        }
        let rate = passed as f32 / self.spec.cases.len() as f32;
        let syntax = f32::from(report["syntax"] == true);
        let interface = f32::from(report["interface"] == true);
        report["generated_tokens"] = json!(rollouts[0].tokens.len());
        report["completion_bytes"] = json!(bytes.len());
        report["evaluation_ms"] = json!(start.elapsed().as_secs_f64() * 1000.0);
        report["provenance"] = self.spec.provenance();
        ennx_wire::json::pretty_writer(
            File::create(path.join("reward.json")).map_err(|e| e.to_string())?,
            &report,
        )
        .map_err(|e| e.to_string())?;
        eprintln!(
            "ENNX_CODE_EVALUATION syntax={syntax} interface={interface} passed={passed}/{}",
            self.spec.cases.len()
        );
        native_vector(
            vec![rate],
            &[vec![syntax], vec![interface], vec![rate]],
            &["syntax_valid", "entrypoint_defined", "execution_pass_rate"],
        )
    }
}

fn write_request(mut input: std::process::ChildStdin, request: &Value) -> Result<(), String> {
    ennx_wire::json::to_writer(&mut input, request).map_err(|e| e.to_string())?;
    input.flush().map_err(|e| e.to_string())
}

fn failed_report(reason: &str, count: usize) -> Value {
    json!({"syntax":false,"interface":false,"passed":0,"total":count,
        "status":reason,"cases":[]})
}

#[repr(C)]
#[derive(Default)]
struct TaskUsage {
    virtual_size: u64,
    resident_size: u64,
    times: [u64; 4],
    counters: [i32; 12],
}

#[link(name = "proc")]
unsafe extern "C" {
    fn proc_pidinfo(
        pid: i32,
        flavor: i32,
        arg: u64,
        buffer: *mut std::ffi::c_void,
        size: i32,
    ) -> i32;
}

fn resident_bytes(pid: u32) -> Option<u64> {
    let mut usage = TaskUsage::default();
    let size = std::mem::size_of::<TaskUsage>() as i32;
    // Darwin PROC_PIDTASKINFO=4; TaskUsage matches sys/proc_info.h (96 bytes).
    // The supplied pointer is writable for exactly the reported buffer size.
    let written = unsafe {
        proc_pidinfo(
            pid as i32,
            4,
            0,
            (&mut usage as *mut TaskUsage).cast(),
            size,
        )
    };
    (written == size).then_some(usage.resident_size)
}

fn sandbox_profile(interpreter: &Path) -> Result<String, String> {
    let output = Command::new(interpreter)
        .args([
            "-I",
            "-S",
            "-B",
            "-c",
            "import json,sys; print(json.dumps(sys.path + [sys.base_prefix]))",
        ])
        .env_clear()
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("cannot resolve Python runtime paths".into());
    }
    let mut roots: Vec<String> =
        ennx_wire::json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
    roots.push(
        interpreter
            .parent()
            .ok_or("interpreter has no parent")?
            .to_string_lossy()
            .into_owned(),
    );
    roots.extend([
        "/System/Library".into(),
        "/System/Volumes/Preboot/Cryptexes/OS".into(),
        "/Library/Apple/System/Library".into(),
        "/usr/lib".into(),
        "/dev/null".into(),
    ]);
    let mut profile =
        String::from("(version 1)(deny default)(allow sysctl-read)(allow file-read-metadata)");
    let binary =
        ennx_wire::json::to_string(&interpreter.to_string_lossy()).map_err(|e| e.to_string())?;
    // dyld opens the root directory as a path anchor; this is not recursive access.
    profile.push_str(&format!("(allow process-exec (literal {binary}))(allow file-write* (literal \"/dev/null\"))(allow file-read* file-map-executable (literal \"/\")"));
    for root in roots {
        let root = std::fs::canonicalize(&root)
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or(root);
        let root = ennx_wire::json::to_string(&root).map_err(|e| e.to_string())?;
        profile.push_str(&format!("(subpath {root})"));
    }
    profile.push(')');
    Ok(profile)
}

// Python is the language under evaluation, not the optimizer or CLI implementation.
// The deny-default OS sandbox is the security boundary; namespace restrictions are not.
const HARNESS: &str = r#"
import ast, contextlib, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
request = json.load(sys.stdin)
report = dict(syntax=False, interface=False, passed=0, total=len(request['inputs']), cases=[], status='syntax_error')
try:
    tree = ast.parse(request['source'], filename='<generated>')
    program = compile(tree, '<generated>', 'exec')
    report['syntax'] = True
    report['interface'] = any(isinstance(n, ast.FunctionDef) and n.name == request['entrypoint'] for n in tree.body)
    if request['execute']:
        namespace = {}
        with open('/dev/null', 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(program, namespace)
            function = namespace.get(request['entrypoint'])
            report['interface'] = callable(function)
            for index, args in enumerate(request['inputs']):
                try:
                    actual = function(*args)
                    json.dumps(actual, allow_nan=False)
                    report['cases'].append(dict(index=index, actual=actual))
                except BaseException as error:
                    report['cases'].append(dict(index=index, passed=False, error=type(error).__name__))
    report['status'] = 'evaluated'
except BaseException as error:
    report['error'] = type(error).__name__
json.dump(report, sys.stdout)
"#;
