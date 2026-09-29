//! Layout algebra and bank-conflict verification for Turing Shared Memory.

/// Swizzle transformation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwizzleMode {
    None,
    /// Pad each row by `pad_elements`
    Pad {
        pad_elements: usize,
    },
    /// Apply XOR swizzle: `(row ^ (col / unit)) % 32`
    Xor128B,
}

/// A 2D tile layout in Shared Memory or Global Memory.
#[derive(Debug, Clone)]
pub struct TileLayout {
    pub rows: usize,
    pub cols: usize,
    pub element_bytes: usize,
    pub swizzle: SwizzleMode,
}

impl TileLayout {
    pub const fn new(rows: usize, cols: usize, element_bytes: usize, swizzle: SwizzleMode) -> Self {
        Self {
            rows,
            cols,
            element_bytes,
            swizzle,
        }
    }

    /// Computes the leading dimension stride in bytes.
    pub fn stride_bytes(&self) -> usize {
        match self.swizzle {
            SwizzleMode::None | SwizzleMode::Xor128B => self.cols * self.element_bytes,
            SwizzleMode::Pad { pad_elements } => (self.cols + pad_elements) * self.element_bytes,
        }
    }

    /// Computes the byte offset for an element at `(row, col)`.
    pub fn byte_offset(&self, row: usize, col: usize) -> usize {
        assert!(
            row < self.rows && col < self.cols,
            "Out-of-bounds tile access"
        );
        match self.swizzle {
            SwizzleMode::None => (row * self.cols + col) * self.element_bytes,
            SwizzleMode::Pad { pad_elements } => {
                let stride = self.cols + pad_elements;
                (row * stride + col) * self.element_bytes
            }
            SwizzleMode::Xor128B => {
                // 128-byte swizzle for Turing 32 banks (each 4 bytes = 128 bytes total bank cycle)
                let base_offset = (row * self.cols + col) * self.element_bytes;
                let bank_row = row & 0x7;
                let col_chunk = (col * self.element_bytes / 16) & 0x7;
                let swizzled_chunk = col_chunk ^ bank_row;
                (base_offset & !0x7F) | (swizzled_chunk * 16) | (base_offset & 0xF)
            }
        }
    }

    /// Total bytes required to store this tile in shared memory.
    pub fn total_bytes(&self) -> usize {
        self.rows * self.stride_bytes()
    }

    /// Verifies whether an access pattern from a warp (32 threads) results in bank conflicts.
    ///
    /// On Turing `sm_75`, there are 32 shared memory banks, 4 bytes each.
    /// If multiple threads in a warp access different words in the same bank,
    /// the hardware serializes the access ($k$-way bank conflict).
    pub fn verify_warp_access(
        &self,
        warp_indices: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<(), BankConflictReport> {
        let mut bank_hits = [0usize; 32];
        let mut count = 0;

        for (row, col) in warp_indices {
            let offset = self.byte_offset(row, col);
            let bank = (offset / 4) % 32;
            bank_hits[bank] += 1;
            count += 1;
        }

        assert_eq!(
            count, 32,
            "Warp access verification expects exactly 32 thread indices"
        );

        let max_hits = *bank_hits.iter().max().unwrap_or(&0);
        if max_hits <= 1 {
            Ok(())
        } else {
            Err(BankConflictReport {
                max_degree: max_hits,
                bank_hits,
            })
        }
    }
}

/// Diagnostic report when a bank conflict is detected.
#[derive(Debug, Clone)]
pub struct BankConflictReport {
    pub max_degree: usize,
    pub bank_hits: [usize; 32],
}

impl std::fmt::Display for BankConflictReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}-way shared memory bank conflict detected! Max accesses to a single bank: {}",
            self.max_degree, self.max_degree
        )
    }
}

impl std::error::Error for BankConflictReport {}
