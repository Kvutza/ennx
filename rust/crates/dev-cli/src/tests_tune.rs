use super::*;

#[test]
fn parses_evaluation_result() {
    assert!(std::mem::size_of::<Evaluation>() > 0);
    let evaluation: Evaluation =
        parse_result("RESULT 0.75 311 751632384 12.5 1 2 3 1 1\n").unwrap();
    assert_eq!(evaluation.reward, 0.75);
    assert_eq!(evaluation.tensors, 311);
    assert_eq!(evaluation.elements, 751_632_384);
    assert_eq!(evaluation.perturb_ms, 12.5);
    assert!(parse_result("RESULT broken").is_err());
    assert!(parse_result("RESULT NaN 311 751632384 12.5 1 2 3 1 1").is_err());
    assert!(parse_result("RESULT 0.75 0 0 12.5 1 2 3 1 1").is_err());
    assert!(parse_result("RESULT 0.75 311 751632384 -1.0 1 2 3 1 1").is_err());
    assert!(parse_result("RESULT 0.75 311 751632384 12.5 2 2 3 1 1").is_err());
    assert!(parse_result("RESULT 0.75 311 751632384 12.5 1 4 3 1 1").is_err());
    assert!(parse_field::<usize>("bad", "count").is_err());
}

#[test]
fn validates_tune_config() {
    assert!(std::mem::size_of::<TuneConfig>() > 0);
    let config = TuneConfig {
        melville: PathBuf::from("melville"),
        workspace: PathBuf::from("workspace"),
        task: PathBuf::from("task.md"),
        verifier: "true".into(),
        reset: "true".into(),
        model: "qwen-diffusion".into(),
        turns: 1,
        rounds: 1,
        candidates: 2,
        seed: 7,
        radius: 0.01,
        history: 8,
    };
    assert_eq!(config.model, "qwen-diffusion");
    assert_eq!(config.history, 8);
    assert!(config.validate().is_ok());

    let mut invalid = config;
    invalid.candidates = 0;
    assert!(invalid.validate().is_err());
}
