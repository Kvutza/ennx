use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn copy_tree(source: &Path, output: &Path) -> io::Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source = entry.path();
        let destination = output.join(entry.file_name());
        if source.is_dir() {
            fs::create_dir_all(&destination)?;
            copy_tree(&source, &destination)?;
        } else {
            fs::copy(source, destination)?;
        }
    }
    Ok(())
}

fn runtime_library(name: &str) -> bool {
    name == "libfaiss.so"
        || name == "libgfortran.so.5"
        || name == "libgcc_s.so.1"
        || name == "libgomp.so.1"
        || name == "libquadmath.so.0"
        || name == "libstdc++.so.6"
        || name.starts_with("libstdc++.so.6.")
        || (name.starts_with("libopenblas") && name.contains(".so"))
}

fn pixi_prefix(environment: &str) -> io::Result<PathBuf> {
    let cwd = env::current_dir()?;
    cwd.ancestors()
        .map(|root| root.join(".pixi").join("envs").join(environment))
        .find(|prefix| prefix.is_dir())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("no .pixi/envs/{environment} above {}", cwd.display()),
            )
        })
}

fn main() -> io::Result<()> {
    let mut args = env::args_os().skip(1);
    let output = PathBuf::from(args.next().expect("missing output path"));
    let environment = args.next().expect("missing Pixi environment");
    assert!(args.next().is_none(), "unexpected argument");
    let prefix = pixi_prefix(&environment.to_string_lossy())?;

    let include = output.join("include").join("faiss");
    fs::create_dir_all(&include)?;
    copy_tree(&prefix.join("include").join("faiss"), &include)?;

    let lib_output = output.join("lib");
    fs::create_dir_all(&lib_output)?;
    for entry in fs::read_dir(prefix.join("lib"))? {
        let entry = entry?;
        let name = entry.file_name();
        if runtime_library(&name.to_string_lossy()) {
            fs::copy(entry.path(), lib_output.join(name))?;
        }
    }
    Ok(())
}
