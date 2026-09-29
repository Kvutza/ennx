//! Low-level instruction emission methods for PTX kernel builder.

use crate::ir::PtxKernelBuilder;
use crate::reg::{MmaAccumulator, Reg32, Reg64, RegPair, RegPred, RegQuad};

pub trait PtxEmitter {
    fn raw(&mut self, line: impl AsRef<str>);

    // Special register access
    fn get_tid_x(&mut self, dst: Reg32) {
        self.raw(format!("mov.u32 %r{}, %tid.x;", dst.0));
    }

    fn get_ntid_x(&mut self, dst: Reg32) {
        self.raw(format!("mov.u32 %r{}, %ntid.x;", dst.0));
    }

    fn get_ctaid_x(&mut self, dst: Reg32) {
        self.raw(format!("mov.u32 %r{}, %ctaid.x;", dst.0));
    }

    fn get_ctaid_y(&mut self, dst: Reg32) {
        self.raw(format!("mov.u32 %r{}, %ctaid.y;", dst.0));
    }

    // Parameter loading
    fn ld_param_u64(&mut self, dst: Reg64, param_name: &str) {
        self.raw(format!("ld.param.u64 %rd{}, [{param_name}];", dst.0));
    }

    fn ld_param_u32(&mut self, dst: Reg32, param_name: &str) {
        self.raw(format!("ld.param.u32 %r{}, [{param_name}];", dst.0));
    }

    // Vectorized 128-bit global memory load (Optimal on T4)
    fn ld_global_v4_u32(&mut self, dst: RegQuad, ptr: Reg64) {
        self.raw(format!(
            "ld.global.nc.v4.u32 {{%r{}, %r{}, %r{}, %r{}}}, [%rd{}];",
            dst.0.0, dst.1.0, dst.2.0, dst.3.0, ptr.0
        ));
    }

    // Vectorized 128-bit global memory store
    fn st_global_v4_u32(&mut self, ptr: Reg64, src: RegQuad) {
        self.raw(format!(
            "st.global.v4.u32 [%rd{}], {{%r{}, %r{}, %r{}, %r{}}};",
            ptr.0, src.0.0, src.1.0, src.2.0, src.3.0
        ));
    }

    // 128-bit shared memory load/store
    fn ld_shared_v4_u32(&mut self, dst: RegQuad, smem_offset: Reg32) {
        self.raw(format!(
            "ld.shared.v4.u32 {{%r{}, %r{}, %r{}, %r{}}}, [smem + %r{}];",
            dst.0.0, dst.1.0, dst.2.0, dst.3.0, smem_offset.0
        ));
    }

    fn st_shared_v4_u32(&mut self, smem_offset: Reg32, src: RegQuad) {
        self.raw(format!(
            "st.shared.v4.u32 [smem + %r{}], {{%r{}, %r{}, %r{}, %r{}}};",
            smem_offset.0, src.0.0, src.1.0, src.2.0, src.3.0
        ));
    }

    // Arithmetic
    fn add_u64(&mut self, dst: Reg64, a: Reg64, b: Reg64) {
        self.raw(format!("add.u64 %rd{}, %rd{}, %rd{};", dst.0, a.0, b.0));
    }

    fn add_u32(&mut self, dst: Reg32, a: Reg32, b: Reg32) {
        self.raw(format!("add.u32 %r{}, %r{}, %r{};", dst.0, a.0, b.0));
    }

    fn mul_lo_u32(&mut self, dst: Reg32, a: Reg32, b: Reg32) {
        self.raw(format!("mul.lo.u32 %r{}, %r{}, %r{};", dst.0, a.0, b.0));
    }

    fn mad_lo_u32(&mut self, dst: Reg32, a: Reg32, b: Reg32, c: Reg32) {
        self.raw(format!(
            "mad.lo.u32 %r{}, %r{}, %r{}, %r{};",
            dst.0, a.0, b.0, c.0
        ));
    }

    fn shl_imm_u32(&mut self, dst: Reg32, src: Reg32, shift: u32) {
        self.raw(format!("shl.b32 %r{}, %r{}, {shift};", dst.0, src.0));
    }

    fn cvt_u64_u32(&mut self, dst: Reg64, src: Reg32) {
        self.raw(format!("cvt.u64.u32 %rd{}, %r{};", dst.0, src.0));
    }

    // Turing Tensor Core: mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    fn mma_sync_m16n8k8(
        &mut self,
        dst: MmaAccumulator,
        a_frag: RegPair, // 2 x 32-bit registers containing 4 x FP16
        b_frag: Reg32,   // 1 x 32-bit register containing 2 x FP16
        c_acc: MmaAccumulator,
    ) {
        self.raw(format!(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {{%f{}, %f{}, %f{}, %f{}}}, {{%r{}, %r{}}}, {{%r{}}}, {{%f{}, %f{}, %f{}, %f{}}};",
            dst.0 .0, dst.1 .0, dst.2 .0, dst.3 .0,
            a_frag.0 .0, a_frag.1 .0,
            b_frag.0,
            c_acc.0 .0, c_acc.1 .0, c_acc.2 .0, c_acc.3 .0
        ));
    }

    // Turing Tensor Core: INT8 mma.sync.aligned.m8n8k16.s32.s8.s8.s32
    fn mma_sync_m8n8k16_int8(
        &mut self,
        dst: RegPair,
        a_frag: RegPair,
        b_frag: Reg32,
        c_acc: RegPair,
    ) {
        self.raw(format!(
            "mma.sync.aligned.m8n8k16.s32.s8.s8.s32 {{%r{}, %r{}}}, {{%r{}, %r{}}}, {{%r{}}}, {{%r{}, %r{}}};",
            dst.0 .0, dst.1 .0,
            a_frag.0 .0, a_frag.1 .0,
            b_frag.0,
            c_acc.0 .0, c_acc.1 .0
        ));
    }

    // Barrier synchronization
    fn bar_sync(&mut self, barrier_id: u32) {
        self.raw(format!("bar.sync {barrier_id};"));
    }

    // Branching & predicates
    fn setp_ge_u32(&mut self, pred: RegPred, a: Reg32, b: Reg32) {
        self.raw(format!("setp.ge.u32 %p{}, %r{}, %r{};", pred.0, a.0, b.0));
    }

    fn setp_lt_u32(&mut self, pred: RegPred, a: Reg32, b: Reg32) {
        self.raw(format!("setp.lt.u32 %p{}, %r{}, %r{};", pred.0, a.0, b.0));
    }

    fn branch_if(&mut self, pred: RegPred, target_label: &str) {
        self.raw(format!("@%p{} bra {target_label};", pred.0));
    }

    fn branch_if_not(&mut self, pred: RegPred, target_label: &str) {
        self.raw(format!("@!%p{} bra {target_label};", pred.0));
    }

    fn exit(&mut self) {
        self.raw("ret;");
    }
}

impl PtxEmitter for PtxKernelBuilder {
    fn raw(&mut self, line: impl AsRef<str>) {
        self.raw(line);
    }
}
