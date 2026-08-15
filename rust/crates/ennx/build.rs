use std::path::{Path, PathBuf};

fn has_faiss(dir: &Path) -> bool {
    ["libfaiss.dylib", "libfaiss.so"]
        .iter()
        .any(|name| dir.join(name).exists())
}

fn candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = std::env::var_os("FAISS_LIB_DIR") {
        paths.push(path.into());
    }
    if let Some(prefix) = std::env::var_os("CONDA_PREFIX") {
        paths.push(PathBuf::from(prefix).join("lib"));
    }
    if cfg!(target_os = "macos") {
        paths.extend([
            PathBuf::from("/opt/homebrew/opt/faiss/lib"),
            PathBuf::from("/usr/local/opt/faiss/lib"),
        ]);
    } else if cfg!(target_os = "linux") {
        paths.extend([
            PathBuf::from("/usr/lib/x86_64-linux-gnu"),
            PathBuf::from("/usr/lib/aarch64-linux-gnu"),
            PathBuf::from("/usr/local/lib"),
        ]);
    }
    paths
}

fn include_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = std::env::var_os("FAISS_INCLUDE_DIR") {
        paths.push(path.into());
    }
    if let Some(prefix) = std::env::var_os("CONDA_PREFIX") {
        paths.push(PathBuf::from(prefix).join("include"));
    }
    paths.extend([
        PathBuf::from("/opt/homebrew/opt/faiss/include"),
        PathBuf::from("/usr/local/opt/faiss/include"),
        PathBuf::from("/usr/local/include"),
        PathBuf::from("/usr/include"),
    ]);
    paths
}

fn main() {
    println!("cargo:rerun-if-env-changed=FAISS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=FAISS_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-changed=src/faiss_bridge.cpp");
    if std::env::var_os("DOCS_RS").is_some() {
        return;
    }
    let lib = candidates()
        .into_iter()
        .find(|candidate| has_faiss(candidate))
        .expect("Faiss library was not found; set FAISS_LIB_DIR");
    let include = include_candidates()
        .into_iter()
        .find(|candidate| candidate.join("faiss/IndexFlat.h").exists())
        .expect("Faiss headers were not found; set FAISS_INCLUDE_DIR");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/faiss_bridge.cpp")
        .include(include)
        .flag_if_supported("-std=c++17")
        .warnings(false)
        .compile("ennx_faiss_bridge");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=faiss");
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
