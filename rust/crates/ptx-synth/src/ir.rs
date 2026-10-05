//! Symbolic intermediate representation and emitter for PTX targeting Turing (`sm_75`).

use std::fmt::Write;

pub use crate::reg::{MmaAccumulator, Reg32, Reg64, RegF32, RegPair, RegPred, RegQuad, TargetArch};

/// Virtual register allocator with strict tracking of register bank counts.
#[derive(Debug, Default, Clone)]
pub struct RegAlloc {
    next_r: u32,
    next_rd: u32,
    next_f: u32,
    next_p: u32,
}

impl RegAlloc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc_r32(&mut self) -> Reg32 {
        let r = Reg32(self.next_r);
        self.next_r += 1;
        r
    }

    pub fn alloc_r64(&mut self) -> Reg64 {
        let rd = Reg64(self.next_rd);
        self.next_rd += 1;
        rd
    }

    pub fn alloc_f32(&mut self) -> RegF32 {
        let f = RegF32(self.next_f);
        self.next_f += 1;
        f
    }

    pub fn alloc_pred(&mut self) -> RegPred {
        let p = RegPred(self.next_p);
        self.next_p += 1;
        p
    }

    pub fn alloc_pair(&mut self) -> RegPair {
        RegPair(self.alloc_r32(), self.alloc_r32())
    }

    pub fn alloc_quad(&mut self) -> RegQuad {
        RegQuad(
            self.alloc_r32(),
            self.alloc_r32(),
            self.alloc_r32(),
            self.alloc_r32(),
        )
    }

    pub fn alloc_mma_accum(&mut self) -> MmaAccumulator {
        MmaAccumulator(
            self.alloc_f32(),
            self.alloc_f32(),
            self.alloc_f32(),
            self.alloc_f32(),
        )
    }

    /// Total 32-bit equivalent hardware registers allocated.
    /// Note: 64-bit registers count as two 32-bit registers.
    pub fn total_32bit_regs(&self) -> u32 {
        self.next_r + (self.next_rd * 2) + self.next_f
    }
}

/// A kernel entry parameter definition.
#[derive(Debug, Clone)]
pub struct KernelParam {
    pub name: String,
    pub ptx_type: &'static str, // e.g. ".u64", ".u32", ".f32"
}

/// Strongly-typed PTX code builder with hardware budget assertions.
pub struct PtxKernelBuilder {
    pub name: String,
    pub arch: TargetArch,
    pub max_threads_per_block: (u32, u32, u32),
    pub max_regs: Option<u32>,
    pub smem_bytes: usize,
    pub params: Vec<KernelParam>,
    pub alloc: RegAlloc,
    pub body: Vec<String>,
    label_counter: u32,
}

impl PtxKernelBuilder {
    pub fn new(name: impl Into<String>, arch: TargetArch) -> Self {
        Self {
            name: name.into(),
            arch,
            max_threads_per_block: (128, 1, 1),
            max_regs: Some(128), // Default max occupancy target on T4
            smem_bytes: 0,
            params: Vec::new(),
            alloc: RegAlloc::new(),
            body: Vec::new(),
            label_counter: 0,
        }
    }

    pub fn set_threads_per_block(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.max_threads_per_block = (x, y, z);
        self
    }

    pub fn set_max_regs(&mut self, max_regs: u32) -> &mut Self {
        self.max_regs = Some(max_regs);
        self
    }

    pub fn set_shared_memory_bytes(&mut self, bytes: usize) -> &mut Self {
        let max_smem = match self.arch {
            TargetArch::Sm75 => 65536,
            TargetArch::Sm80 => 167936,
            TargetArch::Sm90 => 232448,
        };
        assert!(
            bytes <= max_smem,
            "Shared memory requested ({bytes} B) exceeds {} limit ({max_smem} B)!",
            self.arch.ptx_target_str()
        );
        self.smem_bytes = bytes;
        self
    }

    pub fn add_param(&mut self, name: impl Into<String>, ptx_type: &'static str) -> &mut Self {
        self.params.push(KernelParam {
            name: name.into(),
            ptx_type,
        });
        self
    }

    pub fn new_label(&mut self, prefix: &str) -> String {
        let label = format!("{prefix}_{}", self.label_counter);
        self.label_counter += 1;
        label
    }

    pub fn mark_label(&mut self, label: &str) {
        self.body.push(format!("{label}:"));
    }

    pub fn raw(&mut self, line: impl AsRef<str>) {
        self.body.push(format!("    {}", line.as_ref()));
    }

    /// Emits the verified, completed PTX code string.
    pub fn emit(&self) -> String {
        let total_regs = self.alloc.total_32bit_regs();
        if let Some(limit) = self.max_regs {
            assert!(
                total_regs <= limit,
                "Register budget violated! Allocated {total_regs} 32-bit registers, limit is {limit}."
            );
        }

        let mut ptx = String::with_capacity(4096);
        writeln!(
            ptx,
            "// Synthesized by ptx-synth for {}",
            self.arch.ptx_target_str()
        )
        .unwrap();
        writeln!(ptx, ".version {}", self.arch.ptx_version_str()).unwrap();
        writeln!(ptx, ".target {}", self.arch.ptx_target_str()).unwrap();
        writeln!(ptx, ".address_size 64\n").unwrap();

        // Entry header
        write!(ptx, ".visible .entry {}(\n", self.name).unwrap();
        for (i, p) in self.params.iter().enumerate() {
            let comma = if i + 1 < self.params.len() { "," } else { "" };
            writeln!(ptx, "    .param {} {}{}", p.ptx_type, p.name, comma).unwrap();
        }
        writeln!(ptx, ")").unwrap();

        // Block & occupancy directives
        writeln!(
            ptx,
            ".maxntid {}, {}, {}",
            self.max_threads_per_block.0,
            self.max_threads_per_block.1,
            self.max_threads_per_block.2
        )
        .unwrap();

        if let Some(limit) = self.max_regs {
            writeln!(ptx, ".maxnreg {limit}").unwrap();
        }

        writeln!(ptx, "{{").unwrap();

        // Register declarations
        if self.alloc.next_p > 0 {
            writeln!(ptx, "    .reg .pred %p<{}>;", self.alloc.next_p).unwrap();
        }
        if self.alloc.next_r > 0 {
            writeln!(ptx, "    .reg .b32 %r<{}>;", self.alloc.next_r).unwrap();
        }
        if self.alloc.next_rd > 0 {
            writeln!(ptx, "    .reg .b64 %rd<{}>;", self.alloc.next_rd).unwrap();
        }
        if self.alloc.next_f > 0 {
            writeln!(ptx, "    .reg .f32 %f<{}>;", self.alloc.next_f).unwrap();
        }

        // Shared memory allocation
        if self.smem_bytes > 0 {
            writeln!(ptx, "    .shared .align 16 .b8 smem[{}];", self.smem_bytes).unwrap();
        }

        writeln!(ptx).unwrap();

        // Body
        for line in &self.body {
            writeln!(ptx, "{line}").unwrap();
        }

        writeln!(ptx, "}}").unwrap();
        ptx
    }
}
