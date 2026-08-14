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
        || name == "faiss.dll"
        || name == "libgfortran.so.5"
        || name == "libgcc_s.so.1"
        || name == "libgomp.so.1"
        || name == "libquadmath.so.0"
        || name == "libstdc++.so.6"
        || name.starts_with("libstdc++.so.6.")
        || (name.starts_with("libopenblas") && name.contains(".so"))
        || (name.starts_with("openblas") && name.ends_with(".dll"))
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

fn copy_windows_python_import_lib(
    prefix: &Path,
    native: &Path,
    lib_output: &Path,
    python_abi: &str,
) -> io::Result<()> {
    let candidates = [prefix.join("libs"), native.join("libs"), native.join("lib")];
    for directory in candidates {
        if !directory.is_dir() {
            continue;
        }
        let mut libraries = Vec::new();
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("python") && name.ends_with(".lib") && name != "pythonXY.lib" {
                libraries.push((name, entry.path()));
            }
        }
        libraries.sort_by_key(|(name, _)| {
            if name.eq_ignore_ascii_case(&format!("python{}.lib", &python_abi[2..])) {
                0
            } else if name
                .strip_prefix("python")
                .and_then(|rest| rest.strip_suffix(".lib"))
                .is_some_and(|version| version.chars().all(|c| c.is_ascii_digit()))
            {
                1
            } else {
                2
            }
        });
        if let Some((name, path)) = libraries.into_iter().next() {
            fs::copy(&path, lib_output.join(&name))?;
            fs::copy(path, lib_output.join("pythonXY.lib"))?;
            return Ok(());
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "no Windows Python import library found under {}",
            prefix.display()
        ),
    ))
}

fn main() -> io::Result<()> {
    let mut args = env::args_os().skip(1);
    let output = PathBuf::from(args.next().expect("missing output path"));
    let environment = args.next().expect("missing Pixi environment");
    let python_abi = args
        .next()
        .expect("missing Python ABI")
        .into_string()
        .expect("Python ABI must be UTF-8");
    assert!(args.next().is_none(), "unexpected argument");
    let prefix = pixi_prefix(&environment.to_string_lossy())?;
    let windows = cfg!(target_os = "windows");
    let native = if windows {
        prefix.join("Library")
    } else {
        prefix.clone()
    };

    let include = output.join("include").join("faiss");
    fs::create_dir_all(&include)?;
    copy_tree(&native.join("include").join("faiss"), &include)?;

    let lib_output = output.join("lib");
    fs::create_dir_all(&lib_output)?;
    let lib_input = native.join("lib");
    for entry in fs::read_dir(&lib_input)? {
        let entry = entry?;
        if windows && entry.file_name().to_string_lossy() == "faiss.lib" {
            fs::copy(entry.path(), lib_output.join(entry.file_name()))?;
        }
    }
    if windows {
        copy_windows_python_import_lib(&prefix, &native, &lib_output, &python_abi)?;
    }
    let bin_input = if windows {
        native.join("bin")
    } else {
        native.join("lib")
    };
    for entry in fs::read_dir(bin_input)? {
        let entry = entry?;
        let name = entry.file_name();
        if runtime_library(&name.to_string_lossy()) {
            fs::copy(entry.path(), lib_output.join(name))?;
        }
    }
    Ok(())
}
