//! ALICE-SIMD — Shared SIMD, branchless, and fast-math primitives
//!
//! Common performance primitives extracted from the ALICE ecosystem:
//! - `AlignedVec<T>`: 32-byte aligned vector for SIMD loads
//! - `BitMask64`: 64-bit packed bitmask with branchless operations
//! - Branchless `select`, `min`, `max`, `clamp`, `abs` for f32
//! - Fast math: `fast_rcp`, `fast_inv_sqrt`, `fma`, `lerp`
//! - `fnv1a`: deterministic hash for content deduplication

#![no_std]
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::inline_always
)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod aligned;
pub mod bitmask;
pub mod branchless;
pub mod fast_math;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod hash;

pub use aligned::AlignedVec;
pub use bitmask::{BitMask64, ComparisonMask, SetBitIterator};
pub use branchless::{
    branchless_abs, branchless_clamp, branchless_max, branchless_min, select_f32,
};
pub use fast_math::{
    batch_fma, batch_mul_scalar, deg_to_rad, distance_squared, fast_inv_sqrt, fast_rcp, fast_sqrt,
    fma, fma_chain, length_squared, lerp, normalize, Reciprocals, RECIPROCALS,
};
pub use hash::fnv1a;

/// SIMD width for the current platform (in f32 lanes).
pub const SIMD_WIDTH: usize = if cfg!(target_feature = "avx2") {
    8
} else {
    4 // NEON or SSE
};

/// Align a count up to the next multiple of `SIMD_WIDTH`.
#[inline(always)]
#[must_use]
pub const fn align_up(n: usize) -> usize {
    (n + SIMD_WIDTH - 1) & !(SIMD_WIDTH - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up_zero() {
        assert_eq!(align_up(0), 0);
    }

    #[test]
    fn test_align_up_exact() {
        assert_eq!(align_up(SIMD_WIDTH), SIMD_WIDTH);
    }

    #[test]
    fn test_align_up_one_over() {
        assert_eq!(align_up(SIMD_WIDTH + 1), SIMD_WIDTH * 2);
    }

    #[test]
    fn test_align_up_one() {
        // 1 should round up to SIMD_WIDTH
        assert_eq!(align_up(1), SIMD_WIDTH);
    }

    #[test]
    fn test_simd_width_is_power_of_two() {
        assert!(SIMD_WIDTH.is_power_of_two());
    }
}
