//! Independent repetitions share data but derive their own random streams.

use ennx::config::ConfigOverrides;
use ennx_wire::json::{Value, json};
use std::fs::{self, File};
use std::path::Path;

pub(super) fn run(config: &ConfigOverrides, directory: &Path) -> Result<(), String> {
    let dataset = config
        .dataset()
        .ok_or("generated pretraining requires a resolved dataset")?;
    let mut results = Vec::new();
    for rep in 0..config.reps() {
        let output = if config.reps() == 1 {
            directory.to_owned()
        } else {
            let output = directory.join(format!("rep-{:03}", rep + 1));
            fs::create_dir(&output).map_err(|e| e.to_string())?;
            output
        };
        let repetition = settings(config, rep);
        ennx_wire::json::pretty_writer(
            File::create(output.join("repetition.json")).map_err(|e| e.to_string())?,
            &json!({"rep":rep+1,"reps":config.reps(),"selection":config.selection.unwrap_or_default(),
                "model_seed":repetition.model_seed,"proposal_seed":repetition.proposal_seed,
                "acquisition_seed":repetition.acquisition_seed,"sampling_seed":repetition.sample_seeded(0)}),
        ).map_err(|e| e.to_string())?;
        eprintln!("ENNX_GENERATION_REP rep={} reps={}", rep + 1, config.reps());
        ennx::experimental::run_generated(&repetition, dataset, &output)?;
        let result: Value = ennx_wire::json::from_reader(
            File::open(output.join("result.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        results.push(json!({"rep":rep+1,"artifact":output,"result":result}));
    }
    if config.reps() > 1 {
        ennx_wire::json::pretty_writer(
            File::create(directory.join("result.json")).map_err(|e| e.to_string())?,
            &json!({"schema":"ennx.generation_repetitions.v1","status":"completed",
                "teacher_forcing":false,"generation_in_loop":true,"repetitions":results}),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn settings(config: &ConfigOverrides, rep: u32) -> ConfigOverrides {
    let mut repetition = config.clone();
    repetition.reps = Some(1);
    repetition.model_seed = Some(config.model_seeded(rep));
    repetition.proposal_seed = Some(config.proposal_seeded(rep));
    repetition.acquisition_seed = Some(config.acquisition_seeded(rep));
    if let Some(generation) = &mut repetition.generation {
        generation.seed = Some(config.sample_seeded(rep));
    }
    repetition
}
