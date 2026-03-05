//! Branchless `f32` primitives using bit-level manipulation.
//!
//! All functions are `#[inline(always)]` and avoid conditional branches
//! entirely, enabling the compiler / processor to avoid pipeline flushes on
//! heavily data-dependent paths such as SIMD classification loops.

// ── select_f32 ─────────────────────────────────────────────────────────────

/// Returns `a` if `condition` is `true`, `b` otherwise — without branching.
///
/// Equivalent to a hardware `CMOV` (conditional move) or `VBLENDVPS`
/// instruction.  The bit-mask approach compiles to a `cmov` on `x86_64` and
/// `csel` on aarch64.
///
/// # Examples
/// ```
/// # use alice_simd::select_f32;
/// assert_eq!(select_f32(true,  1.0, 2.0), 1.0);
/// assert_eq!(select_f32(false, 1.0, 2.0), 2.0);
/// ```
#[inline(always)]
#[must_use]
pub fn select_f32(condition: bool, a: f32, b: f32) -> f32 {
    // -(true as i32) == -1 == 0xFFFF_FFFF; -(false as i32) == 0
    let mask = -i32::from(condition) as u32;
    f32::from_bits((a.to_bits() & mask) | (b.to_bits() & !mask))
}

// ── branchless_min ─────────────────────────────────────────────────────────

/// Returns the smaller of `a` and `b` without branching.
///
/// # Examples
/// ```
/// # use alice_simd::branchless_min;
/// assert_eq!(branchless_min(3.0_f32, 5.0), 3.0);
/// assert_eq!(branchless_min(5.0_f32, 3.0), 3.0);
/// ```
#[inline(always)]
#[must_use]
pub fn branchless_min(a: f32, b: f32) -> f32 {
    select_f32(a < b, a, b)
}

// ── branchless_max ─────────────────────────────────────────────────────────

/// Returns the larger of `a` and `b` without branching.
///
/// # Examples
/// ```
/// # use alice_simd::branchless_max;
/// assert_eq!(branchless_max(3.0_f32, 5.0), 5.0);
/// assert_eq!(branchless_max(5.0_f32, 3.0), 5.0);
/// ```
#[inline(always)]
#[must_use]
pub fn branchless_max(a: f32, b: f32) -> f32 {
    select_f32(a > b, a, b)
}

// ── branchless_clamp ───────────────────────────────────────────────────────

/// Clamps `x` to `[lo, hi]` without branching.
///
/// # Examples
/// ```
/// # use alice_simd::branchless_clamp;
/// assert_eq!(branchless_clamp(0.5_f32, 0.0, 1.0), 0.5);
/// assert_eq!(branchless_clamp(-1.0_f32, 0.0, 1.0), 0.0);
/// assert_eq!(branchless_clamp(2.0_f32, 0.0, 1.0), 1.0);
/// ```
#[inline(always)]
#[must_use]
pub fn branchless_clamp(x: f32, lo: f32, hi: f32) -> f32 {
    branchless_min(branchless_max(x, lo), hi)
}

// ── branchless_abs ─────────────────────────────────────────────────────────

/// Returns the absolute value of `x` without branching.
///
/// Clears the sign bit via bitwise AND, which is identical to what
/// `f32::abs()` compiles to on modern targets.
///
/// # Examples
/// ```
/// # use alice_simd::branchless_abs;
/// assert_eq!(branchless_abs(-3.5_f32), 3.5);
/// assert_eq!(branchless_abs(3.5_f32), 3.5);
/// assert_eq!(branchless_abs(0.0_f32), 0.0);
/// ```
#[inline(always)]
#[must_use]
pub const fn branchless_abs(x: f32) -> f32 {
    // Clear the sign bit (bit 31)
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

// ── branchless_sign ────────────────────────────────────────────────────────

/// Returns `1.0` if `x >= 0.0`, `-1.0` otherwise — without branching.
#[inline(always)]
#[must_use]
pub const fn branchless_sign(x: f32) -> f32 {
    // Preserve the sign bit, set the exponent+mantissa to 1.0 (0x3F800000)
    let sign_bit = x.to_bits() & 0x8000_0000;
    f32::from_bits(sign_bit | 0x3F80_0000)
}

/// Returns `1.0` if `x > 0.0`, `-1.0` if `x < 0.0`, `0.0` if `x == 0.0`.
#[inline(always)]
#[must_use]
pub fn branchless_signum(x: f32) -> f32 {
    // (x > 0) - (x < 0) mapped to f32
    let pos = i32::from(x > 0.0_f32);
    let neg = i32::from(x < 0.0_f32);
    (pos - neg) as f32
}

/// Returns `true` if `x` is negative (including negative zero) — branchless.
#[inline(always)]
#[must_use]
pub const fn is_negative(x: f32) -> bool {
    // Sign bit is the MSB
    (x.to_bits() >> 31) != 0
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ── select_f32 ──────────────────────────────────────────────────────────

    #[test]
    fn test_select_true() {
        assert_eq!(select_f32(true, 1.0_f32, 2.0), 1.0_f32);
    }

    #[test]
    fn test_select_false() {
        assert_eq!(select_f32(false, 1.0_f32, 2.0), 2.0_f32);
    }

    #[test]
    fn test_select_equal_values() {
        assert_eq!(select_f32(true, 7.0_f32, 7.0), 7.0_f32);
        assert_eq!(select_f32(false, 7.0_f32, 7.0), 7.0_f32);
    }

    #[test]
    fn test_select_negative() {
        assert_eq!(select_f32(true, -5.0_f32, 5.0), -5.0_f32);
        assert_eq!(select_f32(false, -5.0_f32, 5.0), 5.0_f32);
    }

    // ── branchless_min ──────────────────────────────────────────────────────

    #[test]
    fn test_min_left_smaller() {
        assert_eq!(branchless_min(1.0_f32, 2.0), 1.0_f32);
    }

    #[test]
    fn test_min_right_smaller() {
        assert_eq!(branchless_min(5.0_f32, 3.0), 3.0_f32);
    }

    #[test]
    fn test_min_equal() {
        assert_eq!(branchless_min(4.0_f32, 4.0), 4.0_f32);
    }

    #[test]
    fn test_min_negative() {
        assert_eq!(branchless_min(-1.0_f32, -2.0), -2.0_f32);
    }

    // ── branchless_max ──────────────────────────────────────────────────────

    #[test]
    fn test_max_left_larger() {
        assert_eq!(branchless_max(9.0_f32, 3.0), 9.0_f32);
    }

    #[test]
    fn test_max_right_larger() {
        assert_eq!(branchless_max(1.0_f32, 8.0), 8.0_f32);
    }

    #[test]
    fn test_max_equal() {
        assert_eq!(branchless_max(3.0_f32, 3.0), 3.0_f32);
    }

    #[test]
    fn test_max_negative() {
        assert_eq!(branchless_max(-3.0_f32, -7.0), -3.0_f32);
    }

    // ── branchless_clamp ────────────────────────────────────────────────────

    #[test]
    fn test_clamp_within() {
        assert_eq!(branchless_clamp(0.5_f32, 0.0, 1.0), 0.5_f32);
    }

    #[test]
    fn test_clamp_below() {
        assert_eq!(branchless_clamp(-1.0_f32, 0.0, 1.0), 0.0_f32);
    }

    #[test]
    fn test_clamp_above() {
        assert_eq!(branchless_clamp(2.0_f32, 0.0, 1.0), 1.0_f32);
    }

    #[test]
    fn test_clamp_at_boundary() {
        assert_eq!(branchless_clamp(0.0_f32, 0.0, 1.0), 0.0_f32);
        assert_eq!(branchless_clamp(1.0_f32, 0.0, 1.0), 1.0_f32);
    }

    // ── branchless_abs ──────────────────────────────────────────────────────

    #[test]
    fn test_abs_negative() {
        assert_eq!(branchless_abs(-3.5_f32), 3.5_f32);
    }

    #[test]
    fn test_abs_positive() {
        assert_eq!(branchless_abs(3.5_f32), 3.5_f32);
    }

    #[test]
    fn test_abs_zero() {
        assert_eq!(branchless_abs(0.0_f32), 0.0_f32);
    }

    #[test]
    fn test_abs_large() {
        assert_eq!(branchless_abs(-1_000_000.0_f32), 1_000_000.0_f32);
    }

    // ── extras ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sign_positive() {
        assert_eq!(branchless_sign(5.0_f32), 1.0_f32);
    }

    #[test]
    fn test_sign_negative() {
        assert_eq!(branchless_sign(-5.0_f32), -1.0_f32);
    }

    #[test]
    fn test_signum() {
        assert_eq!(branchless_signum(3.0_f32), 1.0_f32);
        assert_eq!(branchless_signum(-3.0_f32), -1.0_f32);
        assert_eq!(branchless_signum(0.0_f32), 0.0_f32);
    }

    #[test]
    fn test_is_negative() {
        assert!(is_negative(-1.0_f32));
        assert!(!is_negative(1.0_f32));
        assert!(!is_negative(0.0_f32));
    }
}
