use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

struct Args {
    src_dir: PathBuf,
    out: PathBuf,
    package: String,
    version: String,
    platform_tag: String,
    python_abi: String,
    python_requires: String,
    readme: PathBuf,
    extension_suffix: String,
}

fn args() -> Args {
    let mut values = env::args().skip(1);
    let mut get = |expected: &str| {
        assert_eq!(values.next().as_deref(), Some(expected));
        values
            .next()
            .unwrap_or_else(|| panic!("missing {expected}"))
    };
    let result = Args {
        src_dir: get("--src-dir").into(),
        out: get("--out").into(),
        package: get("--package"),
        version: get("--version"),
        platform_tag: get("--platform-tag"),
        python_abi: get("--python-abi"),
        python_requires: get("--python-requires"),
        readme: get("--readme").into(),
        extension_suffix: get("--extension-suffix"),
    };
    assert!(values.next().is_none(), "unexpected argument");
    result
}

fn files_below(root: &Path) -> io::Result<Vec<PathBuf>> {
    fn visit(dir: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                visit(&path, files)?;
            } else {
                files.push(path);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn add_optimizer_fixtures(
    args: &Args,
    wheel_files: &mut BTreeMap<String, Vec<u8>>,
) -> io::Result<()> {
    let fixture_root = args.src_dir.join("tests/fixtures");
    for source in files_below(&fixture_root)? {
        let relative = source.strip_prefix(&fixture_root).unwrap();
        wheel_files.insert(
            format!(
                "{}/turbo/optimizer_fixtures/data/{}",
                args.package,
                relative.to_string_lossy().replace('\\', "/")
            ),
            fs::read(source)?,
        );
    }
    Ok(())
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h = [
        0x6a09e667_u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0_u32; 64];
        for (i, bytes) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(bytes.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (state, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *state = state.wrapping_add(value);
        }
    }
    let mut digest = [0_u8; 32];
    for (slot, value) in digest.chunks_exact_mut(4).zip(h) {
        slot.copy_from_slice(&value.to_be_bytes());
    }
    digest
}

fn base64_url(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let value = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(TABLE[((value >> 18) & 63) as usize] as char);
        out.push(TABLE[((value >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((value >> 6) & 63) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(TABLE[(value & 63) as usize] as char);
        }
    }
    out
}

fn u16_le(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn u32_le(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn zip(entries: &BTreeMap<String, Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let name = name.as_bytes();
        let offset = out.len() as u32;
        let crc = crc32(data);
        u32_le(&mut out, 0x0403_4b50);
        u16_le(&mut out, 20);
        u16_le(&mut out, 0);
        u16_le(&mut out, 0);
        u16_le(&mut out, 0);
        u16_le(&mut out, 0x21);
        u32_le(&mut out, crc);
        u32_le(&mut out, data.len() as u32);
        u32_le(&mut out, data.len() as u32);
        u16_le(&mut out, name.len() as u16);
        u16_le(&mut out, 0);
        out.extend_from_slice(name);
        out.extend_from_slice(data);

        u32_le(&mut central, 0x0201_4b50);
        u16_le(&mut central, 0x0314);
        u16_le(&mut central, 20);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0x21);
        u32_le(&mut central, crc);
        u32_le(&mut central, data.len() as u32);
        u32_le(&mut central, data.len() as u32);
        u16_le(&mut central, name.len() as u16);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0);
        u32_le(&mut central, 0o100644 << 16);
        u32_le(&mut central, offset);
        central.extend_from_slice(name);
    }
    let central_offset = out.len() as u32;
    let central_len = central.len() as u32;
    out.extend_from_slice(&central);
    u32_le(&mut out, 0x0605_4b50);
    u16_le(&mut out, 0);
    u16_le(&mut out, 0);
    u16_le(&mut out, entries.len() as u16);
    u16_le(&mut out, entries.len() as u16);
    u32_le(&mut out, central_len);
    u32_le(&mut out, central_offset);
    u16_le(&mut out, 0);
    out
}

fn main() -> io::Result<()> {
    let args = args();
    let source_package = args.src_dir.join("src").join(&args.package);
    let all_files = files_below(&args.src_dir)?;
    let extensions: Vec<_> = all_files
        .iter()
        .filter(|path| {
            let name = path.file_name().unwrap().to_string_lossy();
            name.starts_with("librust") && name.contains("ennx-py") && name.ends_with(".so")
        })
        .collect();
    assert_eq!(
        extensions.len(),
        1,
        "expected one PyO3 library: {extensions:?}"
    );

    let dist_info = format!("{}-{}.dist-info", args.package, args.version);
    let tag = format!("{0}-{0}-{1}", args.python_abi, args.platform_tag);
    let mut wheel_files = BTreeMap::new();
    for source in files_below(&source_package)? {
        if source.extension().and_then(|value| value.to_str()) == Some("py") {
            let relative = source.strip_prefix(&source_package).unwrap();
            wheel_files.insert(
                format!(
                    "{}/{}",
                    args.package,
                    relative.to_string_lossy().replace('\\', "/")
                ),
                fs::read(source)?,
            );
        }
    }
    add_optimizer_fixtures(&args, &mut wheel_files)?;
    wheel_files.insert(
        format!("{}/ennx_rust{}", args.package, args.extension_suffix),
        fs::read(extensions[0])?,
    );
    if let Some(openmp) = all_files
        .iter()
        .find(|path| path.file_name().and_then(|name| name.to_str()) == Some("libomp.dylib"))
    {
        wheel_files.insert(
            format!("{}/.dylibs/libomp.dylib", args.package),
            fs::read(openmp)?,
        );
    }
    if args.platform_tag.starts_with("manylinux") {
        for library in all_files.iter().filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.contains(".so") && !name.contains("ennx-py") && !name.contains("ennx_py")
                })
        }) {
            let name = library.file_name().unwrap().to_string_lossy();
            wheel_files.insert(
                format!("{}/.dylibs/{name}", args.package),
                fs::read(library)?,
            );
        }
    }
    wheel_files.insert(
        format!("{dist_info}/licenses/LICENSE"),
        fs::read(args.src_dir.join("LICENSE"))?,
    );
    wheel_files.insert(
        format!("{dist_info}/licenses/NOTICE"),
        fs::read(args.src_dir.join("NOTICE"))?,
    );
    let mut metadata = format!(
        "Metadata-Version: 2.3\nName: {}\nVersion: {}\nSummary: Epistemic Nearest Neighbors\nRequires-Python: {}\nRequires-Dist: numpy>=2.1\nRequires-Dist: scipy>=1.11\nProvides-Extra: gp\nRequires-Dist: torch>=2.0; extra == 'gp'\nRequires-Dist: gpytorch>=1.11; extra == 'gp'\nDescription-Content-Type: text/markdown; charset=UTF-8\n\n",
        args.package, args.version, args.python_requires
    )
    .into_bytes();
    metadata.extend(fs::read(args.src_dir.join(args.readme))?);
    wheel_files.insert(format!("{dist_info}/METADATA"), metadata);
    wheel_files.insert(
        format!("{dist_info}/WHEEL"),
        format!("Wheel-Version: 1.0\nGenerator: buck2\nRoot-Is-Purelib: false\nTag: {tag}\n\n")
            .into_bytes(),
    );

    let record_path = format!("{dist_info}/RECORD");
    let mut record = String::new();
    for (path, data) in &wheel_files {
        record.push_str(&format!(
            "{path},sha256={},{}\n",
            base64_url(&sha256(data)),
            data.len()
        ));
    }
    record.push_str(&format!("{record_path},,\n"));
    wheel_files.insert(record_path, record.into_bytes());

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = fs::File::create(args.out)?;
    output.write_all(&zip(&wheel_files))
}
