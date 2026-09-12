use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/flame.cu");
    println!("cargo:rerun-if-changed=native/flame.h");
    for name in [
        "CUDA_HOME",
        "CUDA_PATH",
        "NVCC",
        "ENNX_FLAME_CUDA_ARCH",
        "CARGO_FEATURE_NATIVE_FLAME",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if env::var_os("CARGO_FEATURE_NATIVE_FLAME").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64")
    {
        return;
    }
    let cuda = PathBuf::from(
        env::var_os("CUDA_HOME")
            .or_else(|| env::var_os("CUDA_PATH"))
            .unwrap_or_else(|| "/usr/local/cuda".into()),
    );
    let nvcc = env::var_os("NVCC")
        .map(PathBuf::from)
        .unwrap_or_else(|| cuda.join("bin/nvcc"));
    let arch = env::var("ENNX_FLAME_CUDA_ARCH").unwrap_or_else(|_| "sm_75".into());
    assert!(
        arch.strip_prefix("sm_")
            .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|x| x.is_ascii_digit())),
        "ENNX_FLAME_CUDA_ARCH must be an sm_ architecture, such as sm_75"
    );
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let status = Command::new(&nvcc)
        .args([
            "--lib",
            "-std=c++17",
            "-O3",
            "--fmad=false",
            "-Xcompiler=-fPIC",
        ])
        .arg(format!("-arch={arch}"))
        .arg("native/flame.cu")
        .arg("-o")
        .arg(output.join("libennx_flame_native.a"))
        .status()
        .unwrap_or_else(|error| panic!("Could not run {}: {error}; set CUDA_HOME", nvcc.display()));
    assert!(status.success(), "FLAME CUDA compilation failed");
    println!("cargo:rustc-link-search=native={}", output.display());
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=static=ennx_flame_native");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
