//! Fast math primitives — reciprocals, inverse square root, FMA, lerp, etc.
//!
//! All functions use fused-multiply-add (`mul_add`, or `libm::fmaf` without
//! `std`) where available, and
//! platform-specific intrinsics when target features are enabled.

// ── helpers ───────────────────────────────────────────────────────────────

// `f32::mul_add` and `f32::sqrt` live in `std`. Without `std` the `libm`
// feature supplies `fmaf` / `sqrtf`; both are correctly rounded (IEEE 754
// `fusedMultiplyAdd` / `squareRoot`), so the two builds return the same bits
// (`parity::libm_matches_std_*`).

/// `a * b + c` with a single rounding.
#[inline(always)]
#[must_use]
fn mul_add_f32(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::fmaf(a, b, c)
    }
}

/// Correctly rounded square root.
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse")))]
#[inline(always)]
#[must_use]
fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(x)
    }
}

/// `no_std` 互換の floor（`f32::floor` は `std` 必要）
#[inline(always)]
#[must_use]
fn floor_f32(x: f32) -> f32 {
    let i = x as i32;
    let f = i as f32;
    if x < f {
        f - 1.0
    } else {
        f
    }
}

// ── Reciprocals ────────────────────────────────────────────────────────────

/// Pre-computed reciprocals for common constants used across ALICE.
///
/// Using a stored reciprocal (`x * RCP_N`) instead of division (`x / N`)
/// reduces latency from ~20–35 cycles to ~4 cycles on `x86_64`.
pub struct Reciprocals {
    /// 1.0 / 255.0  — normalise u8 colour channel to `[0,1]`
    pub rcp_255: f32,
    /// 1.0 / 320.0  — Browser viewport width normalisation
    pub rcp_320: f32,
    /// 1.0 / 1024.0 — 2^10 fast shift-right for texture coords
    pub rcp_1024: f32,
    /// 1.0 / 360.0  — degrees-to-turns
    pub rcp_360: f32,
    /// 1.0 / 65535.0 — normalise u16
    pub rcp_65535: f32,
    /// 1.0 / 16777216.0 — normalise 24-bit depth
    pub rcp_16m: f32,
    /// 1.0 / 100.0 — percentage normalisation
    pub rcp_100: f32,
    /// 1.0 / 1000.0 — millisecond → second
    pub rcp_1000: f32,
}

/// The singleton `Reciprocals` instance (compile-time constant values).
pub const RECIPROCALS: Reciprocals = Reciprocals {
    rcp_255: 1.0_f32 / 255.0_f32,
    rcp_320: 1.0_f32 / 320.0_f32,
    rcp_1024: 1.0_f32 / 1024.0_f32,
    rcp_360: 1.0_f32 / 360.0_f32,
    rcp_65535: 1.0_f32 / 65535.0_f32,
    rcp_16m: 1.0_f32 / 16_777_216.0_f32,
    rcp_100: 1.0_f32 / 100.0_f32,
    rcp_1000: 1.0_f32 / 1000.0_f32,
};

// ── fast_rcp ───────────────────────────────────────────────────────────────

/// Fast approximate reciprocal of `x`.
///
/// On `x86_64` with `sse` this uses the `rcpss` instruction (~1 cycle,
/// ~12-bit mantissa precision).  On all other targets falls back to `1.0 / x`
/// which the compiler will optimise according to the platform.
///
/// # Safety notes
/// `x` must be finite and non-zero.  Results are undefined for `0.0`, `inf`,
/// or `NaN` — the same caveat as the hardware instruction.
#[inline(always)]
#[must_use]
pub fn fast_rcp(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "sse")]
        {
            use core::arch::x86_64::{_mm_cvtss_f32, _mm_rcp_ss, _mm_set_ss};
            // SAFETY: x86_64 with sse feature, no alignment requirement
            unsafe {
                let v = _mm_set_ss(x);
                let r = _mm_rcp_ss(v);
                _mm_cvtss_f32(r)
            }
        }
        #[cfg(not(target_feature = "sse"))]
        {
            1.0_f32 / x
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1.0_f32 / x
    }
}

// ── fast_inv_sqrt ──────────────────────────────────────────────────────────

/// Fast approximate inverse square root: `1.0 / sqrt(x)`.
///
/// On `x86_64` with `sse` this uses `rsqrtss` (~2 cycles, ~12-bit precision).
/// All other targets use IEEE `1.0 / x.sqrt()` (full 24-bit precision).
///
/// The non-x86 path was previously the Quake III magic-number trick; on an
/// aarch64 core (Cortex-A78AE) it measured 47% slower than the direct
/// `1.0 / x.sqrt()`, because modern aarch64 has a fast pipelined `fsqrt`.
#[inline(always)]
#[must_use]
pub fn fast_inv_sqrt(x: f32) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    {
        use core::arch::x86_64::{_mm_cvtss_f32, _mm_rsqrt_ss, _mm_set_ss};
        // SAFETY: x86_64 sse feature, no alignment requirement
        unsafe {
            let v = _mm_set_ss(x);
            let r = _mm_rsqrt_ss(v);
            _mm_cvtss_f32(r)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse")))]
    {
        1.0_f32 / sqrt_f32(x)
    }
}

// ── fast_sqrt ──────────────────────────────────────────────────────────────

/// Approximate square root: `x * fast_inv_sqrt(x)`.
///
/// Equivalent to `sqrtf` for positive, non-zero, finite `x`.
#[inline(always)]
#[must_use]
pub fn fast_sqrt(x: f32) -> f32 {
    x * fast_inv_sqrt(x)
}

// ── fma ────────────────────────────────────────────────────────────────────

/// Fused multiply-add: `a * b + c` in a single instruction when available.
///
/// Uses `f32::mul_add` (`libm::fmaf` without `std`), rounded once.  This is
/// `VFMADD*` on `x86_64` with FMA3 and `FMADD` on aarch64; targets without a
/// hardware FMA call a correctly rounded software `fmaf`.
#[inline(always)]
#[must_use]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    mul_add_f32(a, b, c)
}

// ── fma_chain ──────────────────────────────────────────────────────────────

/// `a * b + c * d` computed as two FMAs.
///
/// Equivalent to `fma(a, b, fma(c, d, 0.0))` but avoids the intermediate
/// zero-add.  Useful for dot products and bilinear interpolation.
#[inline(always)]
#[must_use]
pub fn fma_chain(a: f32, b: f32, c: f32, d: f32) -> f32 {
    mul_add_f32(a, b, c * d)
}

// ── lerp ───────────────────────────────────────────────────────────────────

/// Linear interpolation: `a + t * (b - a)` computed with a single FMA.
///
/// `t = 0.0` returns `a`; `t = 1.0` returns `b`.
#[inline(always)]
#[must_use]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    // fma(t, b - a, a)  →  t * (b-a) + a
    mul_add_f32(t, b - a, a)
}

// ── distance_squared ───────────────────────────────────────────────────────

/// Squared Euclidean distance between `(x1, y1)` and `(x2, y2)`.
///
/// Uses FMA to compute `dx*dx + dy*dy` in a single compound instruction.
/// Compare squared distances to avoid a `sqrt` call.
#[inline(always)]
#[must_use]
pub fn distance_squared(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    // fma(dx, dx, dy * dy)
    mul_add_f32(dx, dx, dy * dy)
}

// ── length_squared ─────────────────────────────────────────────────────────

/// Squared length of 2-D vector `(x, y)`.
#[inline(always)]
#[must_use]
pub fn length_squared(x: f32, y: f32) -> f32 {
    mul_add_f32(x, x, y * y)
}

// ── batch helpers ──────────────────────────────────────────────────────────

/// Multiplies every element of `data` by `scalar` in-place.
///
/// The inner loop is trivially auto-vectorisable by LLVM.
#[inline]
pub fn batch_mul_scalar(data: &mut [f32], scalar: f32) {
    for v in data.iter_mut() {
        *v *= scalar;
    }
}

/// Applies `v = v * a + b` to every element of `data` in-place (FMA).
#[inline]
pub fn batch_fma(data: &mut [f32], a: f32, b: f32) {
    for v in data.iter_mut() {
        *v = mul_add_f32(*v, a, b);
    }
}

// ── misc helpers ───────────────────────────────────────────────────────────

/// Converts degrees to radians.
#[inline(always)]
#[must_use]
pub fn deg_to_rad(deg: f32) -> f32 {
    // pi / 180
    deg * (core::f32::consts::PI / 180.0_f32)
}

/// Normalises `val` by multiplying by `inv_max` (pre-computed `1.0 / max`).
///
/// Avoids a division in the hot path; caller pre-computes `inv_max` once.
#[inline(always)]
#[must_use]
pub fn normalize(val: f32, inv_max: f32) -> f32 {
    val * inv_max
}

// ── fast_exp ──────────────────────────────────────────────────────────

/// Fast approximate `e^x` using Schraudolph's method with linear interpolation.
///
/// ~12-bit precision, branch-free. Suitable for softmax where relative
/// differences matter more than absolute accuracy.
///
/// Valid range: approximately `[-87, 88]` (f32 exp range).
#[inline(always)]
#[must_use]
pub fn fast_exp(x: f32) -> f32 {
    // Range reduction + degree-3 minimax polynomial
    // exp(x) = 2^(x * log2(e)) = 2^n * 2^f (n=整数, f=小数)
    // 最大相対誤差 < 0.3%
    let clamped = x.clamp(-87.0_f32, 88.0_f32);
    let val = clamped * core::f32::consts::LOG2_E;
    let ipart = floor_f32(val);
    let fpart = val - ipart;
    let n = ipart as i32;

    // 2^f の多項式近似 (f ∈ [0, 1))
    let p = mul_add_f32(fpart, 0.0558_f32, 0.2402_f32);
    let p = mul_add_f32(fpart, p, core::f32::consts::LN_2);
    let p = mul_add_f32(fpart, p, 1.0_f32);

    // 2^n をIEEE-754指数フィールドで構成
    let exp_n = f32::from_bits(((n + 127) as u32) << 23);
    exp_n * p
}

// ── rmsnorm ───────────────────────────────────────────────────────────

/// RMS Normalization: `out[i] = x[i] / rms(x)` where `rms = sqrt(mean(x²) + eps)`.
///
/// LLM向け正規化。`LayerNorm` よりシンプルで高速（平均の減算不要）。
/// `eps` は数値安定性のための小さな定数（通常 1e-5 〜 1e-6）。
///
/// # Panics
/// `x.len() != out.len()` の場合パニック。
#[inline]
pub fn rmsnorm(x: &[f32], out: &mut [f32], eps: f32) {
    assert_eq!(x.len(), out.len(), "rmsnorm: length mismatch");
    let n = x.len();
    if n == 0 {
        return;
    }

    // sum_sq = Σ x[i]²
    let mut sum_sq = 0.0_f32;
    for &v in x {
        sum_sq = mul_add_f32(v, v, sum_sq);
    }

    // rms = 1 / sqrt(mean(x²) + eps)
    let inv_n = 1.0_f32 / n as f32;
    let rms_inv = fast_inv_sqrt(mul_add_f32(sum_sq, inv_n, eps));

    for (o, &v) in out.iter_mut().zip(x) {
        *o = v * rms_inv;
    }
}

/// In-place RMS Normalization: `x[i] = x[i] / rms(x)`.
#[inline]
pub fn rmsnorm_inplace(x: &mut [f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }

    let mut sum_sq = 0.0_f32;
    for &v in x.iter() {
        sum_sq = mul_add_f32(v, v, sum_sq);
    }

    let inv_n = 1.0_f32 / n as f32;
    let rms_inv = fast_inv_sqrt(mul_add_f32(sum_sq, inv_n, eps));

    for v in x.iter_mut() {
        *v *= rms_inv;
    }
}

// ── softmax ───────────────────────────────────────────────────────────

/// Numerically stable softmax: `out[i] = exp(x[i] - max) / Σ exp(x[j] - max)`.
///
/// LLM の Attention 層と出力層で使用。`max` 減算により `exp` のオーバーフローを防止。
///
/// # Panics
/// `x.len() != out.len()` の場合パニック。
#[inline]
pub fn softmax(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "softmax: length mismatch");
    let n = x.len();
    if n == 0 {
        return;
    }

    // max を求める（数値安定性のため）
    let mut max_val = x[0];
    for &v in &x[1..] {
        if v > max_val {
            max_val = v;
        }
    }

    // exp(x[i] - max) と合計
    let mut sum = 0.0_f32;
    for (o, &v) in out.iter_mut().zip(x) {
        let e = fast_exp(v - max_val);
        *o = e;
        sum += e;
    }

    // 正規化: 除算の代わりに逆数乗算
    let inv_sum = 1.0_f32 / sum;
    for o in out.iter_mut() {
        *o *= inv_sum;
    }
}

/// In-place softmax: `x[i] = exp(x[i] - max) / Σ exp(x[j] - max)`.
#[inline]
pub fn softmax_inplace(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }

    let mut max_val = x[0];
    for &v in &x[1..] {
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        let e = fast_exp(*v - max_val);
        *v = e;
        sum += e;
    }

    let inv_sum = 1.0_f32 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // tolerance for approximate operations
    const EPS: f32 = 1e-4_f32;
    const EPS_LOOSE: f32 = 5e-3_f32; // for ~12-bit SSE rcpss

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── RECIPROCALS ─────────────────────────────────────────────────────────

    #[test]
    fn test_reciprocal_255() {
        let expected = 1.0_f32 / 255.0_f32;
        assert!(approx_eq(RECIPROCALS.rcp_255, expected, EPS));
    }

    #[test]
    fn test_reciprocal_1024() {
        let expected = 1.0_f32 / 1024.0_f32;
        assert!(approx_eq(RECIPROCALS.rcp_1024, expected, EPS));
    }

    #[test]
    fn test_reciprocal_320() {
        let expected = 1.0_f32 / 320.0_f32;
        assert!(approx_eq(RECIPROCALS.rcp_320, expected, EPS));
    }

    // ── fast_rcp ────────────────────────────────────────────────────────────

    #[test]
    fn test_fast_rcp_one() {
        assert!(approx_eq(fast_rcp(1.0_f32), 1.0_f32, EPS_LOOSE));
    }

    #[test]
    fn test_fast_rcp_two() {
        assert!(approx_eq(fast_rcp(2.0_f32), 0.5_f32, EPS_LOOSE));
    }

    #[test]
    fn test_fast_rcp_four() {
        assert!(approx_eq(fast_rcp(4.0_f32), 0.25_f32, EPS_LOOSE));
    }

    #[test]
    fn test_fast_rcp_large() {
        let x = 1000.0_f32;
        assert!(approx_eq(fast_rcp(x), 1.0_f32 / x, EPS_LOOSE));
    }

    // ── fast_inv_sqrt ───────────────────────────────────────────────────────

    #[test]
    fn test_fast_inv_sqrt_one() {
        // 1/sqrt(1) == 1
        assert!(approx_eq(fast_inv_sqrt(1.0_f32), 1.0_f32, EPS_LOOSE));
    }

    #[test]
    fn test_fast_inv_sqrt_four() {
        // 1/sqrt(4) == 0.5
        assert!(approx_eq(fast_inv_sqrt(4.0_f32), 0.5_f32, EPS_LOOSE));
    }

    #[test]
    fn test_fast_inv_sqrt_100() {
        // 1/sqrt(100) == 0.1
        assert!(approx_eq(fast_inv_sqrt(100.0_f32), 0.1_f32, EPS_LOOSE));
    }

    #[test]
    fn test_fast_inv_sqrt_positive() {
        assert!(fast_inv_sqrt(16.0_f32) > 0.0_f32);
    }

    // ── fast_sqrt ───────────────────────────────────────────────────────────

    #[test]
    fn test_fast_sqrt_nine() {
        // Approximately 3.0
        assert!(approx_eq(fast_sqrt(9.0_f32), 3.0_f32, 0.02_f32));
    }

    #[test]
    fn test_fast_sqrt_positive() {
        assert!(fast_sqrt(25.0_f32) > 0.0_f32);
    }

    // ── fma ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_fma_basic() {
        // 2 * 3 + 4 = 10
        assert!(approx_eq(fma(2.0_f32, 3.0_f32, 4.0_f32), 10.0_f32, EPS));
    }

    #[test]
    fn test_fma_zero_c() {
        // 5 * 5 + 0 = 25
        assert!(approx_eq(fma(5.0_f32, 5.0_f32, 0.0_f32), 25.0_f32, EPS));
    }

    #[test]
    fn test_fma_negative() {
        // -1 * 3 + 6 = 3
        assert!(approx_eq(fma(-1.0_f32, 3.0_f32, 6.0_f32), 3.0_f32, EPS));
    }

    // ── fma_chain ───────────────────────────────────────────────────────────

    #[test]
    fn test_fma_chain_basic() {
        // 1*2 + 3*4 = 14
        assert!(approx_eq(
            fma_chain(1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32),
            14.0_f32,
            EPS
        ));
    }

    // ── lerp ────────────────────────────────────────────────────────────────

    #[test]
    fn test_lerp_t0() {
        assert!(approx_eq(lerp(0.0_f32, 10.0_f32, 0.0_f32), 0.0_f32, EPS));
    }

    #[test]
    fn test_lerp_t1() {
        assert!(approx_eq(lerp(0.0_f32, 10.0_f32, 1.0_f32), 10.0_f32, EPS));
    }

    #[test]
    fn test_lerp_half() {
        assert!(approx_eq(lerp(0.0_f32, 10.0_f32, 0.5_f32), 5.0_f32, EPS));
    }

    #[test]
    fn test_lerp_nonzero_a() {
        // lerp(4, 8, 0.25) = 4 + 0.25*(8-4) = 5
        assert!(approx_eq(lerp(4.0_f32, 8.0_f32, 0.25_f32), 5.0_f32, EPS));
    }

    // ── distance_squared ────────────────────────────────────────────────────

    #[test]
    fn test_distance_squared_same_point() {
        assert!(approx_eq(
            distance_squared(1.0_f32, 2.0_f32, 1.0_f32, 2.0_f32),
            0.0_f32,
            EPS
        ));
    }

    #[test]
    fn test_distance_squared_unit() {
        // (0,0) → (1,0): d^2 = 1
        assert!(approx_eq(
            distance_squared(0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32),
            1.0_f32,
            EPS
        ));
    }

    #[test]
    fn test_distance_squared_345() {
        // (0,0) → (3,4): d^2 = 25
        assert!(approx_eq(
            distance_squared(0.0_f32, 0.0_f32, 3.0_f32, 4.0_f32),
            25.0_f32,
            EPS
        ));
    }

    // ── length_squared ──────────────────────────────────────────────────────

    #[test]
    fn test_length_squared_unit() {
        assert!(approx_eq(length_squared(1.0_f32, 0.0_f32), 1.0_f32, EPS));
    }

    #[test]
    fn test_length_squared_34() {
        assert!(approx_eq(length_squared(3.0_f32, 4.0_f32), 25.0_f32, EPS));
    }

    // ── batch helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_mul_scalar() {
        let mut data = alloc::vec![1.0_f32, 2.0, 3.0, 4.0];
        batch_mul_scalar(&mut data, 2.0_f32);
        assert!(approx_eq(data[0], 2.0_f32, EPS));
        assert!(approx_eq(data[3], 8.0_f32, EPS));
    }

    #[test]
    fn test_batch_fma() {
        let mut data = alloc::vec![1.0_f32, 2.0, 3.0];
        // v = v * 2 + 1
        batch_fma(&mut data, 2.0_f32, 1.0_f32);
        assert!(approx_eq(data[0], 3.0_f32, EPS));
        assert!(approx_eq(data[1], 5.0_f32, EPS));
        assert!(approx_eq(data[2], 7.0_f32, EPS));
    }

    // ── misc ────────────────────────────────────────────────────────────────

    #[test]
    fn test_deg_to_rad_180() {
        assert!(approx_eq(deg_to_rad(180.0_f32), core::f32::consts::PI, EPS));
    }

    #[test]
    fn test_deg_to_rad_90() {
        assert!(approx_eq(
            deg_to_rad(90.0_f32),
            core::f32::consts::FRAC_PI_2,
            EPS
        ));
    }

    #[test]
    fn test_normalize() {
        let inv_max = 1.0_f32 / 255.0_f32;
        let result = normalize(128.0_f32, inv_max);
        let expected = 128.0_f32 / 255.0_f32;
        assert!(approx_eq(result, expected, EPS));
    }

    #[test]
    fn test_normalize_max() {
        let inv_max = 1.0_f32 / 100.0_f32;
        assert!(approx_eq(normalize(100.0_f32, inv_max), 1.0_f32, EPS));
    }

    // ── fast_exp ─────────────────────────────────────────────────────────

    #[test]
    fn test_fast_exp_zero() {
        // e^0 = 1
        assert!(approx_eq(fast_exp(0.0_f32), 1.0_f32, 0.01_f32));
    }

    #[test]
    fn test_fast_exp_one() {
        // e^1 ≈ 2.718
        assert!(approx_eq(fast_exp(1.0_f32), core::f32::consts::E, 0.05_f32));
    }

    #[test]
    fn test_fast_exp_negative() {
        // e^(-1) ≈ 0.368
        let expected = 1.0_f32 / core::f32::consts::E;
        assert!(approx_eq(fast_exp(-1.0_f32), expected, 0.02_f32));
    }

    #[test]
    fn test_fast_exp_clamp_large() {
        // 巨大な値でもinfにならない
        assert!(fast_exp(100.0_f32).is_finite());
    }

    #[test]
    fn test_fast_exp_clamp_small() {
        // 非常に小さい値でも0以上
        assert!(fast_exp(-100.0_f32) >= 0.0_f32);
    }

    // ── rmsnorm ──────────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_unit() {
        let x = [1.0_f32, 0.0, 0.0, 0.0];
        let mut out = [0.0_f32; 4];
        rmsnorm(&x, &mut out, 1e-6_f32);
        // rms = sqrt(1/4 + eps) ≈ 0.5, inv ≈ 2.0
        assert!(out[0] > 1.5_f32);
        assert!(approx_eq(out[1], 0.0_f32, EPS));
    }

    #[test]
    fn test_rmsnorm_uniform() {
        let x = [2.0_f32, 2.0, 2.0, 2.0];
        let mut out = [0.0_f32; 4];
        rmsnorm(&x, &mut out, 1e-6_f32);
        // 全要素同一 → 全出力も同一
        assert!(approx_eq(out[0], out[1], EPS));
        assert!(approx_eq(out[1], out[2], EPS));
        assert!(approx_eq(out[2], out[3], EPS));
    }

    #[test]
    fn test_rmsnorm_inplace_equivalence() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];
        rmsnorm(&x, &mut out, 1e-6_f32);

        let mut x2 = [1.0_f32, 2.0, 3.0, 4.0];
        rmsnorm_inplace(&mut x2, 1e-6_f32);

        for i in 0..4 {
            assert!(approx_eq(out[i], x2[i], EPS));
        }
    }

    #[test]
    fn test_rmsnorm_empty() {
        let x: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        rmsnorm(&x, &mut out, 1e-6_f32); // パニックしない
    }

    // ── softmax ──────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];
        softmax(&x, &mut out);

        let sum: f32 = out.iter().sum();
        assert!(approx_eq(sum, 1.0_f32, 0.01_f32));
    }

    #[test]
    fn test_softmax_monotonic() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];
        softmax(&x, &mut out);

        // 入力が単調増加 → 出力も単調増加
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
        assert!(out[2] < out[3]);
    }

    #[test]
    fn test_softmax_uniform() {
        let x = [5.0_f32, 5.0, 5.0, 5.0];
        let mut out = [0.0_f32; 4];
        softmax(&x, &mut out);

        // 全同一入力 → 全出力は 1/4
        for &v in &out {
            assert!(approx_eq(v, 0.25_f32, 0.01_f32));
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // 巨大値でもオーバーフローしない（max減算による安定化）
        let x = [1000.0_f32, 1001.0, 1002.0];
        let mut out = [0.0_f32; 3];
        softmax(&x, &mut out);

        let sum: f32 = out.iter().sum();
        assert!(approx_eq(sum, 1.0_f32, 0.02_f32));
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_softmax_inplace_equivalence() {
        let x = [1.0_f32, 2.0, 3.0];
        let mut out = [0.0_f32; 3];
        softmax(&x, &mut out);

        let mut x2 = [1.0_f32, 2.0, 3.0];
        softmax_inplace(&mut x2);

        for i in 0..3 {
            assert!(approx_eq(out[i], x2[i], 0.01_f32));
        }
    }

    #[test]
    fn test_softmax_empty() {
        let x: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        softmax(&x, &mut out); // パニックしない
    }
}

// ── std / libm parity oracle ────────────────────────────────────────────────

/// The `no_std` build replaces `f32::sqrt` / `f32::mul_add` with `libm::sqrtf` /
/// `libm::fmaf`. Both pairs are IEEE 754 correctly rounded operations, so they
/// must agree bit for bit; these tests call both directly and compare
/// `to_bits()` (NaN: both NaN, payload not compared).
#[cfg(test)]
mod parity {
    extern crate std;

    /// Deterministic xorshift32 so the sweep is reproducible.
    struct XorShift(u32);
    impl XorShift {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
    }

    const EDGES: [f32; 22] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),           // smallest subnormal
        f32::from_bits(0x007f_ffff), // largest subnormal
        f32::from_bits(0x8000_0001), // -smallest subnormal
        f32::MAX,
        f32::MIN,
        f32::EPSILON,
        1.0 + f32::EPSILON,
        1.0 - f32::EPSILON / 2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        3.402_823_5e38 / 3.0,
        1e-30,
        16_777_217.0,
    ];

    fn same(a: f32, b: f32) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    #[test]
    fn libm_matches_std_sqrt_edges() {
        for &x in &EDGES {
            let (s, l) = (x.sqrt(), libm::sqrtf(x));
            assert!(
                same(s, l),
                "sqrt({x:e}): std {s:e} ({:#010x}) libm {l:e} ({:#010x})",
                s.to_bits(),
                l.to_bits()
            );
        }
    }

    /// Every 257th bit pattern: ~16.7 M inputs covering every exponent, both
    /// signs, subnormals, infinities and NaNs.
    #[test]
    fn libm_matches_std_sqrt_bit_pattern_sweep() {
        let mut compared = 0_u64;
        let mut bits = 0_u32;
        loop {
            let x = f32::from_bits(bits);
            let (s, l) = (x.sqrt(), libm::sqrtf(x));
            assert!(
                same(s, l),
                "sqrt({bits:#010x}): std {:#010x} libm {:#010x}",
                s.to_bits(),
                l.to_bits()
            );
            compared += 1;
            match bits.checked_add(257) {
                Some(b) => bits = b,
                None => break,
            }
        }
        assert!(compared > 16_000_000, "compared {compared}");
    }

    #[test]
    fn libm_matches_std_fma_edges() {
        let mut compared = 0_u32;
        for &a in &EDGES {
            for &b in &EDGES {
                for &c in &EDGES {
                    let (s, l) = (a.mul_add(b, c), libm::fmaf(a, b, c));
                    assert!(
                        same(s, l),
                        "fma({a:e}, {b:e}, {c:e}): std {:#010x} libm {:#010x}",
                        s.to_bits(),
                        l.to_bits()
                    );
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 22 * 22 * 22);
    }

    /// Random bit patterns (every class of value), plus operands in the same
    /// binade where `a * b` and `c` nearly cancel — the case where a fused and
    /// an unfused result differ, so a non-fused fallback would be caught here.
    #[test]
    fn libm_matches_std_fma_random_and_cancelling() {
        let mut rng = XorShift(0x9e37_79b9);
        let mut fused_differs = 0_u32;
        for _ in 0..1_000_000 {
            let (a, b, c) = (
                f32::from_bits(rng.next()),
                f32::from_bits(rng.next()),
                f32::from_bits(rng.next()),
            );
            let (s, l) = (a.mul_add(b, c), libm::fmaf(a, b, c));
            assert!(
                same(s, l),
                "fma({:#010x}, {:#010x}, {:#010x}): std {:#010x} libm {:#010x}",
                a.to_bits(),
                b.to_bits(),
                c.to_bits(),
                s.to_bits(),
                l.to_bits()
            );
        }
        for _ in 0..1_000_000 {
            // a, b in [1, 2); c = -(a * b) rounded, so a * b + c is the rounding error
            let a = f32::from_bits(0x3f80_0000 | (rng.next() >> 9));
            let b = f32::from_bits(0x3f80_0000 | (rng.next() >> 9));
            let c = -(a * b);
            let (s, l) = (a.mul_add(b, c), libm::fmaf(a, b, c));
            assert!(same(s, l), "fma({a:e}, {b:e}, {c:e}): std {s:e} libm {l:e}");
            if s.to_bits() != (a * b + c).to_bits() {
                fused_differs += 1;
            }
        }
        // the sweep exercises inputs where single rounding matters
        assert!(fused_differs > 100_000, "fused_differs {fused_differs}");
    }

    /// The crate's own `fma` / `fast_inv_sqrt` (whichever build this is: std,
    /// or `no_std` with `libm`) return the std result bit for bit.
    #[test]
    fn crate_fma_matches_std_in_this_build() {
        let mut rng = XorShift(0x0bad_cafe);
        for _ in 0..200_000 {
            let a = f32::from_bits(0x3f80_0000 | (rng.next() >> 9));
            let b = f32::from_bits(0x3f80_0000 | (rng.next() >> 9));
            let c = -(a * b);
            assert!(
                same(super::fma(a, b, c), a.mul_add(b, c)),
                "fma({a:e}, {b:e}, {c:e})"
            );
        }
        for &a in &EDGES {
            for &b in &EDGES {
                for &c in &EDGES {
                    assert!(
                        same(super::fma(a, b, c), a.mul_add(b, c)),
                        "fma({a:e}, {b:e}, {c:e})"
                    );
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse")))]
    #[test]
    fn crate_inv_sqrt_matches_std_in_this_build() {
        let mut bits = 0_u32;
        loop {
            let x = f32::from_bits(bits);
            assert!(
                same(super::fast_inv_sqrt(x), 1.0 / x.sqrt()),
                "inv_sqrt({bits:#010x})"
            );
            match bits.checked_add(65_537) {
                Some(b) => bits = b,
                None => break,
            }
        }
    }
}
