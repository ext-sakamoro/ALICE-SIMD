//! ALICE-SIMD — Shared SIMD, branchless, and fast-math primitives
//!
//! Common performance primitives extracted from the ALICE ecosystem:
//! - `AlignedVec<T>`: 32-byte aligned vector for SIMD loads
//! - `BitMask64`: 64-bit packed bitmask with branchless operations
//! - Branchless `select`, `min`, `max`, `clamp`, `abs` for f32
//! - Fast math: `fast_rcp`, `fast_inv_sqrt`, `fma`, `lerp`
//! - `fnv1a`: deterministic hash for content deduplication
//!
//! # Example
//!
//! The same example is the first code block of README.md
//! (`scripts/docs_lint.py` checks that the two are identical).
//!
//! ```rust
//! use alice_simd::{branchless_min, fast_exp, fnv1a, rmsnorm, softmax, AlignedVec, BitMask64};
//!
//! // fast exp (< 0.3% relative error)
//! let e = fast_exp(1.0);
//! assert!((e - core::f32::consts::E).abs() < 0.01);
//!
//! // RMSNorm (LLM layer normalization), destination-passing style
//! let x = [1.0_f32, 2.0, 3.0, 4.0];
//! let mut out = [0.0_f32; 4];
//! rmsnorm(&x, &mut out, 1e-6);
//!
//! // numerically stable softmax
//! let logits = [1.0_f32, 2.0, 3.0];
//! let mut probs = [0.0_f32; 3];
//! softmax(&logits, &mut probs);
//! assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-3);
//!
//! // branch-free min
//! assert_eq!(branchless_min(3.0, 2.0), 2.0);
//!
//! // 32-byte aligned buffer for SIMD loads / stores
//! let mut buf = AlignedVec::<f32>::new();
//! buf.push(1.0);
//! assert_eq!(buf.as_ptr() as usize % 32, 0);
//!
//! // 64-bit mask
//! let mask = BitMask64(0xFF).set(10);
//! assert_eq!(mask.count_ones(), 9);
//! assert!(mask.test(10));
//!
//! // deterministic content hash
//! let h = fnv1a(b"alice");
//! assert_eq!(h, fnv1a(b"alice"));
//! ```

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

// `fma` / `sqrt` need either `std` or `libm` (see `fast_math`)
#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("alice-simd needs feature `std` (default) or, for no_std targets, feature `libm`");

pub mod aligned;
pub mod bitmask;
pub mod branchless;
pub mod fast_math;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod hash;
pub mod vec3;

pub use aligned::AlignedVec;
pub use bitmask::{BitMask64, ComparisonMask, SetBitIterator};
pub use branchless::{
    branchless_abs, branchless_clamp, branchless_max, branchless_min, select_f32,
};
pub use fast_math::{
    batch_fma, batch_mul_scalar, deg_to_rad, distance_squared, fast_exp, fast_inv_sqrt, fast_rcp,
    fast_sqrt, fma, fma_chain, length_squared, lerp, normalize, rmsnorm, rmsnorm_inplace, softmax,
    softmax_inplace, Reciprocals, RECIPROCALS,
};
pub use hash::fnv1a;
pub use vec3::{Vec3, Vec4};

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
