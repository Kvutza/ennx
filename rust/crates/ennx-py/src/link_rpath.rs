use std::path::{Path, PathBuf};

pub fn blas_present(dir: &Path) -> bool {
    ["libblas.so", "libopenblas.so", "libopenblas.so.0"]
        .iter()
        .any(|name| dir.join(name).exists())
}

pub fn emit_args() {
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if !cfg!(target_os = "linux") {
        return;
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let lib = PathBuf::from(prefix).join("lib");
        if blas_present(&lib) {
            let p = lib.to_string_lossy();
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{p}");
        }
    }
    for p in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"] {
        if blas_present(Path::new(p)) {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{p}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn blas_dir() {
        let dir = std::env::temp_dir().join(format!("enn_blas_check_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!blas_present(Path::new(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emit_smoke() {
        emit_args();
    }
}
