//! FP16 GEMM recipes. Numerical parity and timing are established on hardware.
use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct TuringGemmConfig {
    pub name: String,
    /// Shared-memory staging depth; each MMA consumes eight K elements.
    pub tile_k: u32,
}

impl Default for TuringGemmConfig {
    fn default() -> Self {
        Self {
            name: "turing_fp16_gemm".into(),
            tile_k: 32,
        }
    }
}

/// Row-major A[M,K] * B[K,N] -> C[M,N], FP16 storage, FP32 accumulation.
/// A block covers 64x16 outputs with eight warps. Partial tiles are zero padded.
pub fn synthesize_turing_fp16_gemm(cfg: &TuringGemmConfig) -> String {
    assert!([8, 16, 32, 64].contains(&cfg.tile_k));
    assert!(
        !cfg.name.is_empty()
            && cfg
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
    );
    let k = cfg.tile_k;
    let mut ptx = format!(
        r#".version 6.5
.target sm_75
.address_size 64
.visible .entry {}(
    .param .u64 a, .param .u64 b, .param .u64 c,
    .param .u32 m, .param .u32 n, .param .u32 k
)
.reqntid 256, 1, 1
{{
.reg .pred %p<4>;
.reg .b16 %h<2>;
.reg .b32 %r<40>;
.reg .b64 %rd<6>;
.reg .f32 %f<4>;
.shared .align 16 .b8 left[{}];
.shared .align 16 .b8 right[{}];
mov.u32 %r38, left;
mov.u32 %r39, right;
ld.param.u64 %rd0, [a];
ld.param.u64 %rd1, [b];
ld.param.u64 %rd2, [c];
ld.param.u32 %r9, [m];
ld.param.u32 %r10, [n];
ld.param.u32 %r11, [k];
mov.u32 %r0, %tid.x;
and.b32 %r1, %r0, 31;
shr.u32 %r2, %r0, 5;
shr.u32 %r3, %r1, 2;
and.b32 %r4, %r1, 3;
shl.b32 %r4, %r4, 1;
mov.u32 %r5, %ctaid.y;
shl.b32 %r5, %r5, 6;
mov.u32 %r6, %ctaid.x;
shl.b32 %r6, %r6, 4;
shr.u32 %r7, %r2, 1;
shl.b32 %r7, %r7, 4;
and.b32 %r8, %r2, 1;
shl.b32 %r8, %r8, 3;
add.u32 %r21, %r7, %r3;
add.u32 %r22, %r21, 8;
add.u32 %r23, %r8, %r4;
add.u32 %r24, %r23, 1;
mov.f32 %f0, 0f00000000;
mov.f32 %f1, 0f00000000;
mov.f32 %f2, 0f00000000;
mov.f32 %f3, 0f00000000;
mov.u32 %r12, 0;
K_LOOP:
setp.ge.u32 %p0, %r12, %r11;
@%p0 bra STORE;
mov.u32 %r13, %r0;
A_LOAD:
div.u32 %r14, %r13, {};
rem.u32 %r15, %r13, {};
add.u32 %r16, %r5, %r14;
add.u32 %r17, %r12, %r15;
setp.lt.u32 %p0, %r16, %r9;
setp.lt.u32 %p1, %r17, %r11;
and.pred %p0, %p0, %p1;
mul.wide.u32 %rd3, %r16, %r11;
cvt.u64.u32 %rd4, %r17;
add.u64 %rd3, %rd3, %rd4;
shl.b64 %rd3, %rd3, 1;
add.u64 %rd3, %rd0, %rd3;
mov.b16 %h0, 0;
@%p0 ld.global.b16 %h0, [%rd3];
shl.b32 %r19, %r13, 1;
add.u32 %r19, %r38, %r19;
st.shared.b16 [%r19], %h0;
add.u32 %r13, %r13, 256;
setp.lt.u32 %p0, %r13, {};
@%p0 bra A_LOAD;
mov.u32 %r13, %r0;
B_LOAD:
setp.ge.u32 %p0, %r13, {};
@%p0 bra LOADED;
shr.u32 %r14, %r13, 4;
and.b32 %r15, %r13, 15;
add.u32 %r16, %r12, %r14;
add.u32 %r17, %r6, %r15;
setp.lt.u32 %p0, %r16, %r11;
setp.lt.u32 %p1, %r17, %r10;
and.pred %p0, %p0, %p1;
mul.wide.u32 %rd3, %r16, %r10;
cvt.u64.u32 %rd4, %r17;
add.u64 %rd3, %rd3, %rd4;
shl.b64 %rd3, %rd3, 1;
add.u64 %rd3, %rd1, %rd3;
mov.b16 %h0, 0;
@%p0 ld.global.b16 %h0, [%rd3];
shl.b32 %r19, %r13, 1;
add.u32 %r19, %r39, %r19;
st.shared.b16 [%r19], %h0;
add.u32 %r13, %r13, 256;
bra B_LOAD;
LOADED:
bar.sync 0;
"#,
        cfg.name,
        64 * k * 2,
        k * 16 * 2,
        k,
        k,
        64 * k,
        k * 16
    );
    for start in (0..k).step_by(8) {
        writeln!(ptx, r#"
mad.lo.u32 %r25, %r21, {k}, %r4;
add.u32 %r25, %r25, {start};
shl.b32 %r25, %r25, 1;
add.u32 %r20, %r38, %r25;
ld.shared.b32 %r27, [%r20];
add.u32 %r25, %r25, {};
add.u32 %r20, %r38, %r25;
ld.shared.b32 %r28, [%r20];
add.u32 %r26, %r4, {start};
shl.b32 %r26, %r26, 4;
add.u32 %r26, %r26, %r8;
add.u32 %r26, %r26, %r3;
shl.b32 %r26, %r26, 1;
add.u32 %r20, %r39, %r26;
ld.shared.b16 %h0, [%r20];
add.u32 %r26, %r26, 32;
add.u32 %r20, %r39, %r26;
ld.shared.b16 %h1, [%r20];
mov.b32 %r29, {{%h0, %h1}};
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {{%f0,%f1,%f2,%f3}}, {{%r27,%r28}}, {{%r29}}, {{%f0,%f1,%f2,%f3}};
"#, 8*k*2).unwrap();
    }
    writeln!(
        ptx,
        "bar.sync 0;\nadd.u32 %r12, %r12, {k};\nbra K_LOOP;\nSTORE:"
    )
    .unwrap();
    for (f, row, col) in [(0, 21, 23), (1, 21, 24), (2, 22, 23), (3, 22, 24)] {
        writeln!(
            ptx,
            r#"
add.u32 %r16, %r5, %r{row};
add.u32 %r17, %r6, %r{col};
setp.lt.u32 %p0, %r16, %r9;
setp.lt.u32 %p1, %r17, %r10;
and.pred %p0, %p0, %p1;
mul.wide.u32 %rd3, %r16, %r10;
cvt.u64.u32 %rd4, %r17;
add.u64 %rd3, %rd3, %rd4;
shl.b64 %rd3, %rd3, 1;
add.u64 %rd3, %rd2, %rd3;
cvt.rn.f16.f32 %h0, %f{f};
@%p0 st.global.b16 [%rd3], %h0;
"#
        )
        .unwrap();
    }
    ptx.push_str("ret;\n}\n");
    ptx
}
