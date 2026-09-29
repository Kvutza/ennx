//! High-performance kernel recipes for NVIDIA Turing (sm_75 / T4).

use std::fmt::Write;

use crate::emit::PtxEmitter;
use crate::ir::{PtxKernelBuilder, TargetArch};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeOp {
    Iadd,
    Imul,
    Xor,
    Shf,
    Fadd,
    Fmul,
}

impl ProbeOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Iadd => "iadd",
            Self::Imul => "imul",
            Self::Xor => "xor",
            Self::Shf => "shf",
            Self::Fadd => "fadd",
            Self::Fmul => "fmul",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::Fadd | Self::Fmul)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeShape {
    Chain,
    Ilp8,
}

impl ProbeShape {
    pub fn name(self) -> &'static str {
        match self {
            Self::Chain => "chain",
            Self::Ilp8 => "ilp8",
        }
    }

    fn lanes(self) -> u32 {
        match self {
            Self::Chain => 1,
            Self::Ilp8 => 8,
        }
    }
}

/// Emits a clock64 microbenchmark for one instruction family.
///
/// A dependent chain measures latency. Eight independent chains expose issue
/// throughput. The operation count is supplied at runtime to prevent ptxas
/// from algebraically replacing a long integer chain with one instruction.
/// A disabled body emits the matching loop/clock/barrier baseline.
#[inline(always)]
fn emit_probe_lane_init(ptx: &mut String, op: ProbeOp, lanes: u32) {
    for lane in 0..lanes {
        if op.is_float() {
            writeln!(ptx, "    cvt.rn.f32.u32 %f{lane}, %r0;").unwrap();
            if lane > 0 {
                writeln!(ptx, "    add.f32 %f{lane}, %f{lane}, {lane}.0;").unwrap();
            }
        } else {
            writeln!(
                ptx,
                "    xor.b32 %r{}, %r0, {};",
                8 + lane,
                0x9e37_u32 * (lane + 1)
            )
            .unwrap();
        }
    }
}

#[inline(always)]
fn emit_probe_loop_body(ptx: &mut String, op: ProbeOp, lanes: u32) {
    for lane in 0..lanes {
        match op {
            ProbeOp::Iadd => {
                writeln!(ptx, "    add.u32 %r{0}, %r{0}, %r0;", 8 + lane).unwrap();
            }
            ProbeOp::Imul => {
                writeln!(ptx, "    mul.lo.u32 %r{0}, %r{0}, %r0;", 8 + lane).unwrap();
            }
            ProbeOp::Xor => {
                writeln!(ptx, "    xor.b32 %r{0}, %r{0}, %r0;", 8 + lane).unwrap();
            }
            ProbeOp::Shf => {
                writeln!(
                    ptx,
                    "    shf.r.wrap.b32 %r{0}, %r{0}, %r{0}, %r0;",
                    8 + lane
                )
                .unwrap();
            }
            ProbeOp::Fadd => {
                writeln!(ptx, "    add.f32 %f{lane}, %f{lane}, 0f3e800000;").unwrap();
            }
            ProbeOp::Fmul => {
                writeln!(ptx, "    mul.f32 %f{lane}, %f{lane}, 0f3f800001;").unwrap();
            }
        }
    }
}

#[inline(always)]
fn emit_probe_reduction(ptx: &mut String, op: ProbeOp, lanes: u32) {
    if op.is_float() {
        writeln!(ptx, "    mov.b32 %r16, %f0;").unwrap();
        for lane in 1..lanes {
            writeln!(ptx, "    mov.b32 %r17, %f{lane};").unwrap();
            writeln!(ptx, "    xor.b32 %r16, %r16, %r17;").unwrap();
        }
    } else {
        writeln!(ptx, "    mov.b32 %r16, %r8;").unwrap();
        for lane in 1..lanes {
            writeln!(ptx, "    xor.b32 %r16, %r16, %r{};", 8 + lane).unwrap();
        }
    }
}

pub fn synthesize_turing_probe(
    name: &str,
    op: ProbeOp,
    shape: ProbeShape,
    enabled: bool,
) -> String {
    let lanes = shape.lanes();
    let mut ptx = String::with_capacity(2048);
    writeln!(
        ptx,
        "// Synthesized T4 {} {} probe",
        op.name(),
        shape.name()
    )
    .unwrap();
    writeln!(ptx, ".version 6.4").unwrap();
    writeln!(ptx, ".target sm_75").unwrap();
    writeln!(ptx, ".address_size 64\n").unwrap();
    writeln!(ptx, ".visible .entry {name}(").unwrap();
    writeln!(ptx, "    .param .u64 output,").unwrap();
    writeln!(ptx, "    .param .u32 seed,").unwrap();
    writeln!(ptx, "    .param .u32 rounds").unwrap();
    writeln!(ptx, ")").unwrap();
    writeln!(ptx, ".maxntid 1, 1, 1").unwrap();
    writeln!(ptx, ".maxnreg 32").unwrap();
    writeln!(ptx, "{{").unwrap();
    writeln!(ptx, "    .reg .pred %p<1>;").unwrap();
    writeln!(ptx, "    .reg .b32 %r<24>;").unwrap();
    writeln!(ptx, "    .reg .b64 %rd<4>;").unwrap();
    writeln!(ptx, "    .reg .f32 %f<8>;").unwrap();
    writeln!(ptx, "    .shared .align 4 .b8 fence[4];\n").unwrap();
    writeln!(ptx, "    ld.param.u64 %rd0, [output];").unwrap();
    writeln!(ptx, "    ld.param.u32 %r0, [seed];").unwrap();
    writeln!(ptx, "    ld.param.u32 %r4, [rounds];").unwrap();
    writeln!(ptx, "    mov.u64 %rd1, %clock64;").unwrap();
    writeln!(ptx, "    mov.b64 {{%r1, %r2}}, %rd1;").unwrap();
    writeln!(ptx, "    xor.b32 %r0, %r0, %r1;").unwrap();

    emit_probe_lane_init(&mut ptx, op, lanes);

    writeln!(ptx, "    mov.u32 %r3, 0;").unwrap();
    writeln!(ptx, "PROBE_LOOP:").unwrap();
    if enabled {
        emit_probe_loop_body(&mut ptx, op, lanes);
    }
    writeln!(ptx, "    add.u32 %r3, %r3, 1;").unwrap();
    writeln!(ptx, "    setp.lt.u32 %p0, %r3, %r4;").unwrap();
    writeln!(ptx, "    @%p0 bra PROBE_LOOP;").unwrap();

    emit_probe_reduction(&mut ptx, op, lanes);

    // The shared store and barrier carry the measured chain dependency into
    // the second clock read. The matching disabled-body kernel removes their
    // cost and the loop overhead from the reported instruction cycles.
    writeln!(ptx, "    st.shared.u32 [fence], %r16;").unwrap();
    writeln!(ptx, "    bar.sync 0;").unwrap();
    writeln!(ptx, "    mov.u64 %rd2, %clock64;").unwrap();
    writeln!(ptx, "    sub.u64 %rd3, %rd2, %rd1;").unwrap();
    writeln!(ptx, "    st.global.u64 [%rd0], %rd3;").unwrap();
    writeln!(ptx, "    st.global.u32 [%rd0 + 8], %r16;").unwrap();
    writeln!(ptx, "    ret;").unwrap();
    writeln!(ptx, "}}").unwrap();
    ptx
}

pub use crate::gemm::{TuringGemmConfig, synthesize_turing_fp16_gemm};

/// Synthesizes an ultra-low-latency 128-bit vectorized elementwise kernel on T4.
pub fn synthesize_turing_vector_scale(name: &str) -> String {
    let mut builder = PtxKernelBuilder::new(name, TargetArch::Sm75);
    builder.set_threads_per_block(256, 1, 1);
    builder.set_max_regs(32); // Assembly and launch resources determine occupancy.

    builder.add_param("input_ptr", ".u64");
    builder.add_param("output_ptr", ".u64");
    builder.add_param("scale", ".f32");
    builder.add_param("n_elements", ".u32");

    let tid = builder.alloc.alloc_r32();
    let ntid = builder.alloc.alloc_r32();
    let cta_x = builder.alloc.alloc_r32();
    let global_idx = builder.alloc.alloc_r32();
    let byte_offset = builder.alloc.alloc_r64();

    let in_ptr = builder.alloc.alloc_r64();
    let out_ptr = builder.alloc.alloc_r64();
    let n_elems = builder.alloc.alloc_r32();
    let pred = builder.alloc.alloc_pred();

    builder.get_tid_x(tid);
    builder.get_ntid_x(ntid);
    builder.get_ctaid_x(cta_x);

    builder.ld_param_u64(in_ptr, "input_ptr");
    builder.ld_param_u64(out_ptr, "output_ptr");
    builder.ld_param_u32(n_elems, "n_elements");
    let scale_val = builder.alloc.alloc_f32();
    builder.raw(format!("ld.param.f32 %f{}, [scale];", scale_val.0));

    // global_idx = cta_x * ntid + tid
    builder.mad_lo_u32(global_idx, cta_x, ntid, tid);

    // Multiply by 4 (each thread processes 4 x 32-bit floats = 16 bytes)
    let elem_idx = builder.alloc.alloc_r32();
    builder.shl_imm_u32(elem_idx, global_idx, 2);
    builder.setp_ge_u32(pred, elem_idx, n_elems);
    let exit_label = builder.new_label("EXIT");
    builder.branch_if(pred, &exit_label);

    // Compute byte offset: elem_idx * 4 = global_idx * 16 bytes
    let byte_off_32 = builder.alloc.alloc_r32();
    builder.shl_imm_u32(byte_off_32, global_idx, 4);
    builder.cvt_u64_u32(byte_offset, byte_off_32);

    // Vectorized 128-bit global load: ld.global.nc.v4.f32
    let curr_in = builder.alloc.alloc_r64();
    builder.add_u64(curr_in, in_ptr, byte_offset);
    let f0 = builder.alloc.alloc_f32();
    let f1 = builder.alloc.alloc_f32();
    let f2 = builder.alloc.alloc_f32();
    let f3 = builder.alloc.alloc_f32();
    builder.raw(format!(
        "ld.global.nc.v4.f32 {{%f{}, %f{}, %f{}, %f{}}}, [%rd{}];",
        f0.0, f1.0, f2.0, f3.0, curr_in.0
    ));

    // Multiply each element by scale
    builder.raw(format!(
        "mul.f32 %f{}, %f{}, %f{};",
        f0.0, f0.0, scale_val.0
    ));
    builder.raw(format!(
        "mul.f32 %f{}, %f{}, %f{};",
        f1.0, f1.0, scale_val.0
    ));
    builder.raw(format!(
        "mul.f32 %f{}, %f{}, %f{};",
        f2.0, f2.0, scale_val.0
    ));
    builder.raw(format!(
        "mul.f32 %f{}, %f{}, %f{};",
        f3.0, f3.0, scale_val.0
    ));

    // Vectorized 128-bit global store: st.global.v4.f32
    let curr_out = builder.alloc.alloc_r64();
    builder.add_u64(curr_out, out_ptr, byte_offset);
    builder.raw(format!(
        "st.global.v4.f32 [%rd{}], {{%f{}, %f{}, %f{}, %f{}}};",
        curr_out.0, f0.0, f1.0, f2.0, f3.0
    ));

    builder.mark_label(&exit_label);
    builder.exit();

    builder.emit()
}

/// Synthesizes a high-bandwidth Turing sm_75 fused Rademacher coordinate perturbation kernel.
/// Reads base coordinates in 128-bit vectors, hashes (seed, elem_idx) to generate ±radius steps,
/// and writes perturbed coordinates via 128-bit vector stores.
#[inline(always)]
fn emit_fused_rademacher_hash_elements(
    builder: &mut PtxKernelBuilder,
    elem_idx: crate::reg::Reg32,
    seed_lo: crate::reg::Reg32,
    seed_hi: crate::reg::Reg32,
    radius: crate::reg::RegF32,
    neg_radius: crate::reg::RegF32,
    f_elems: [crate::reg::RegF32; 4],
) {
    let e_idx = builder.alloc.alloc_r32();
    let h = builder.alloc.alloc_r32();
    let mul_res = builder.alloc.alloc_r32();
    let c1 = builder.alloc.alloc_r32();
    let c2 = builder.alloc.alloc_r32();
    let c3 = builder.alloc.alloc_r32();
    let h_shift = builder.alloc.alloc_r32();
    let bit_val = builder.alloc.alloc_r32();
    let bit = builder.alloc.alloc_pred();
    let step = builder.alloc.alloc_f32();

    builder.raw(format!("mov.u32 %r{}, 0x9e3779b9;", c1.0));
    builder.raw(format!("mov.u32 %r{}, 0x7feb352d;", c2.0));
    builder.raw(format!("mov.u32 %r{}, 0x846ca68b;", c3.0));

    for (i, f_elem) in f_elems.into_iter().enumerate() {
        if i == 0 {
            builder.raw(format!("mov.u32 %r{}, %r{};", e_idx.0, elem_idx.0));
        } else {
            builder.raw(format!("add.u32 %r{}, %r{}, {};", e_idx.0, elem_idx.0, i));
        }
        builder.mul_lo_u32(mul_res, e_idx, c1);
        builder.raw(format!(
            "xor.b32 %r{}, %r{}, %r{};",
            h.0, seed_lo.0, mul_res.0
        ));
        builder.raw(format!("shr.u32 %r{}, %r{}, 16;", h_shift.0, h.0));
        builder.raw(format!("xor.b32 %r{}, %r{}, %r{};", h.0, h.0, h_shift.0));
        builder.mul_lo_u32(h, h, c2);
        builder.raw(format!("xor.b32 %r{}, %r{}, %r{};", h.0, h.0, seed_hi.0));
        builder.mul_lo_u32(h, h, c3);
        builder.raw(format!("shr.u32 %r{}, %r{}, 15;", h_shift.0, h.0));
        builder.raw(format!("xor.b32 %r{}, %r{}, %r{};", h.0, h.0, h_shift.0));
        builder.raw(format!("and.b32 %r{}, %r{}, 1;", bit_val.0, h.0));
        builder.raw(format!("setp.ne.u32 %p{}, %r{}, 0;", bit.0, bit_val.0));
        builder.raw(format!(
            "selp.f32 %f{}, %f{}, %f{}, %p{};",
            step.0, radius.0, neg_radius.0, bit.0
        ));
        builder.raw(format!(
            "add.f32 %f{}, %f{}, %f{};",
            f_elem.0, f_elem.0, step.0
        ));
    }
}

struct RademacherParams {
    in_ptr: crate::reg::Reg64,
    out_ptr: crate::reg::Reg64,
    radius: crate::reg::RegF32,
    neg_radius: crate::reg::RegF32,
    seed_lo: crate::reg::Reg32,
    seed_hi: crate::reg::Reg32,
    n_elems: crate::reg::Reg32,
}

#[inline(always)]
fn setup_turing_fused_rademacher_params(builder: &mut PtxKernelBuilder) -> RademacherParams {
    builder.set_threads_per_block(256, 1, 1);
    builder.set_max_regs(48);

    builder.add_param("base_ptr", ".u64");
    builder.add_param("out_ptr", ".u64");
    builder.add_param("radius", ".f32");
    builder.add_param("seed", ".u64");
    builder.add_param("n_elements", ".u32");

    let in_ptr = builder.alloc.alloc_r64();
    let out_ptr = builder.alloc.alloc_r64();
    let radius = builder.alloc.alloc_f32();
    let neg_radius = builder.alloc.alloc_f32();
    let seed = builder.alloc.alloc_r64();
    let seed_lo = builder.alloc.alloc_r32();
    let seed_hi = builder.alloc.alloc_r32();
    let n_elems = builder.alloc.alloc_r32();

    builder.ld_param_u64(in_ptr, "base_ptr");
    builder.ld_param_u64(out_ptr, "out_ptr");
    builder.raw(format!("ld.param.f32 %f{}, [radius];", radius.0));
    builder.raw(format!("neg.f32 %f{}, %f{};", neg_radius.0, radius.0));
    builder.raw(format!("ld.param.u64 %rd{}, [seed];", seed.0));
    builder.raw(format!(
        "mov.b64 {{%r{}, %r{}}}, %rd{};",
        seed_lo.0, seed_hi.0, seed.0
    ));
    builder.ld_param_u32(n_elems, "n_elements");

    RademacherParams {
        in_ptr,
        out_ptr,
        radius,
        neg_radius,
        seed_lo,
        seed_hi,
        n_elems,
    }
}

/// Synthesizes a high-bandwidth Turing sm_75 fused Rademacher coordinate perturbation kernel.
/// Reads base coordinates in 128-bit vectors, hashes (seed, elem_idx) to generate ±radius steps,
/// and writes perturbed coordinates via 128-bit vector stores.
pub fn synthesize_turing_fused_rademacher(name: &str) -> String {
    let mut builder = PtxKernelBuilder::new(name, TargetArch::Sm75);
    let p = setup_turing_fused_rademacher_params(&mut builder);

    let tid = builder.alloc.alloc_r32();
    let ntid = builder.alloc.alloc_r32();
    let cta_x = builder.alloc.alloc_r32();
    let global_idx = builder.alloc.alloc_r32();

    builder.get_tid_x(tid);
    builder.get_ntid_x(ntid);
    builder.get_ctaid_x(cta_x);

    // global_idx = cta_x * ntid + tid
    builder.mad_lo_u32(global_idx, cta_x, ntid, tid);

    // Each thread processes 4 elements (128-bit vector)
    let elem_idx = builder.alloc.alloc_r32();
    builder.shl_imm_u32(elem_idx, global_idx, 2);
    let pred = builder.alloc.alloc_pred();
    builder.setp_ge_u32(pred, elem_idx, p.n_elems);
    let exit_label = builder.new_label("EXIT");
    builder.branch_if(pred, &exit_label);

    let byte_off_32 = builder.alloc.alloc_r32();
    builder.shl_imm_u32(byte_off_32, global_idx, 4);
    let byte_offset = builder.alloc.alloc_r64();
    builder.cvt_u64_u32(byte_offset, byte_off_32);

    let curr_in = builder.alloc.alloc_r64();
    builder.add_u64(curr_in, p.in_ptr, byte_offset);
    let f0 = builder.alloc.alloc_f32();
    let f1 = builder.alloc.alloc_f32();
    let f2 = builder.alloc.alloc_f32();
    let f3 = builder.alloc.alloc_f32();
    builder.raw(format!(
        "ld.global.nc.v4.f32 {{%f{}, %f{}, %f{}, %f{}}}, [%rd{}];",
        f0.0, f1.0, f2.0, f3.0, curr_in.0
    ));

    let f_elems = [f0, f1, f2, f3];
    emit_fused_rademacher_hash_elements(
        &mut builder,
        elem_idx,
        p.seed_lo,
        p.seed_hi,
        p.radius,
        p.neg_radius,
        f_elems,
    );

    let curr_out = builder.alloc.alloc_r64();
    builder.add_u64(curr_out, p.out_ptr, byte_offset);
    builder.raw(format!(
        "st.global.v4.f32 [%rd{}], {{%f{}, %f{}, %f{}, %f{}}};",
        curr_out.0, f0.0, f1.0, f2.0, f3.0
    ));

    builder.mark_label(&exit_label);
    builder.exit();

    builder.emit()
}
