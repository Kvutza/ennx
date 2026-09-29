use ptx_synth::{
    TuringGemmConfig, synthesize_turing_fp16_gemm, synthesize_turing_fused_rademacher,
    synthesize_turing_vector_scale,
};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let target = args.get(1).map(|s| s.as_str()).unwrap_or("gemm");

    match target {
        "vector" => {
            println!("{}", synthesize_turing_vector_scale("turing_vector_scale"));
        }
        "rademacher" => {
            println!(
                "{}",
                synthesize_turing_fused_rademacher("turing_fused_rademacher")
            );
        }
        "hierarchy-parity" => {
            let Some(out_path) = args.get(2).map(std::path::PathBuf::from) else {
                eprintln!("usage: ptx-synth-bin hierarchy-parity OUTPUT");
                std::process::exit(2);
            };
            match ptx_synth::hierarchy::run_hierarchy_parity_check(&out_path) {
                Ok(json) => {
                    println!("HIERARCHY_PARITY ok=true artifact={}", out_path.display());
                    println!("{json}");
                }
                Err(err) => {
                    eprintln!("HIERARCHY_PARITY ok=false error={err}");
                    std::process::exit(1);
                }
            }
        }
        "gemm" => {
            let config = TuringGemmConfig::default();
            println!("{}", synthesize_turing_fp16_gemm(&config));
        }
        _ => {
            eprintln!("unknown PTX recipe: {target}");
            std::process::exit(2);
        }
    }
}
