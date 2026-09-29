#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Leaf {
    pub(super) key: u64,
    pub(super) offset: u64,
    pub(super) length: u64,
    pub(super) scale: f32,
    pub(super) weight: f32,
    pub(super) address: u32,
    pub(super) pad: u32,
}
