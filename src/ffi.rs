//! C FFI for ALICE-SIMD
//!
//! Provides 15 `extern "C"` functions for Unity / UE5 / native integration.
//!
//! License: see LICENSE
//! Author: Moroya Sakamoto

use alloc::boxed::Box;

use crate::aligned::AlignedVec;
use crate::bitmask::BitMask64;
use crate::branchless;
use crate::fast_math;

// ── AlignedVec<f32> (5) ─────────────────────────────────────────────

/// 空の `AlignedVec<f32>` を作成する。
///
/// # Safety
///
/// 戻り値は `alice_aligned_vec_destroy` で解放すること。
#[no_mangle]
pub unsafe extern "C" fn alice_aligned_vec_new() -> *mut AlignedVec<f32> {
    Box::into_raw(Box::new(AlignedVec::new()))
}

/// `AlignedVec<f32>` に値を追加する。
///
/// # Safety
///
/// `vec` は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_aligned_vec_push(vec: *mut AlignedVec<f32>, val: f32) {
    if !vec.is_null() {
        (*vec).push(val);
    }
}

/// `AlignedVec<f32>` の指定インデックスの値を返す。
///
/// # Safety
///
/// `vec` は有効なポインタであること。`idx` は範囲内であること。
/// 範囲外の場合は `0.0` を返す。
#[no_mangle]
pub unsafe extern "C" fn alice_aligned_vec_get(vec: *const AlignedVec<f32>, idx: u32) -> f32 {
    if vec.is_null() {
        return 0.0;
    }
    let v = &*vec;
    if (idx as usize) < v.len() {
        v[idx as usize]
    } else {
        0.0
    }
}

/// `AlignedVec<f32>` の要素数を返す。
///
/// # Safety
///
/// `vec` は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_aligned_vec_len(vec: *const AlignedVec<f32>) -> u32 {
    if vec.is_null() {
        return 0;
    }
    (*vec).len() as u32
}

/// `AlignedVec<f32>` を解放する。
///
/// # Safety
///
/// `vec` は `alice_aligned_vec_new` で取得したポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_aligned_vec_destroy(vec: *mut AlignedVec<f32>) {
    if !vec.is_null() {
        drop(Box::from_raw(vec));
    }
}

// ── BitMask64 (5) ───────────────────────────────────────────────────

/// 指定値で `BitMask64` を作成する。
///
/// # Safety
///
/// 常に安全。
#[no_mangle]
pub extern "C" fn alice_bitmask_new(bits: u64) -> u64 {
    BitMask64(bits).0
}

/// ビット `i` をセットした新しいマスクを返す。
///
/// # Safety
///
/// `i` は 0..63 の範囲であること。
#[no_mangle]
pub extern "C" fn alice_bitmask_set(mask: u64, i: u32) -> u64 {
    if i >= 64 {
        return mask;
    }
    BitMask64(mask).set(i).0
}

/// ビット `i` がセットされているか返す。セット=1, 非セット=0。
///
/// # Safety
///
/// `i` は 0..63 の範囲であること。
#[no_mangle]
pub extern "C" fn alice_bitmask_test(mask: u64, i: u32) -> u8 {
    if i >= 64 {
        return 0;
    }
    u8::from(BitMask64(mask).test(i))
}

/// セットされているビット数を返す。
///
/// # Safety
///
/// 常に安全。
#[no_mangle]
pub extern "C" fn alice_bitmask_count_ones(mask: u64) -> u32 {
    BitMask64(mask).count_ones()
}

/// ビットANDを返す。
///
/// # Safety
///
/// 常に安全。
#[no_mangle]
pub extern "C" fn alice_bitmask_and(a: u64, b: u64) -> u64 {
    BitMask64(a).and(BitMask64(b)).0
}

// ── branchless (3) ──────────────────────────────────────────────────

/// ブランチレスselect: condition!=0 なら `a`、そうでなければ `b` を返す。
///
/// # Safety
///
/// 常に安全。
#[no_mangle]
pub extern "C" fn alice_select_f32(condition: u8, a: f32, b: f32) -> f32 {
    branchless::select_f32(condition != 0, a, b)
}

/// ブランチレスmin。
///
/// # Safety
///
/// 常に安全。
#[no_mangle]
pub extern "C" fn alice_branchless_min(a: f32, b: f32) -> f32 {
    branchless::branchless_min(a, b)
}

/// ブランチレスmax。
///
/// # Safety
///
/// 常に安全。
#[no_mangle]
pub extern "C" fn alice_branchless_max(a: f32, b: f32) -> f32 {
    branchless::branchless_max(a, b)
}

// ── fast_math (2) ───────────────────────────────────────────────────

/// 線形補間: `a + t * (b - a)`。
///
/// # Safety
///
/// 常に安全。
#[no_mangle]
pub extern "C" fn alice_lerp(a: f32, b: f32, t: f32) -> f32 {
    fast_math::lerp(a, b, t)
}

/// Fused multiply-add: `a * b + c`。
///
/// # Safety
///
/// 常に安全。
#[no_mangle]
pub extern "C" fn alice_fma(a: f32, b: f32, c: f32) -> f32 {
    fast_math::fma(a, b, c)
}

// ── テスト ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_aligned_vec_lifecycle() {
        unsafe {
            let vec = alice_aligned_vec_new();
            assert!(!vec.is_null());
            assert_eq!(alice_aligned_vec_len(vec), 0);

            alice_aligned_vec_push(vec, 1.0);
            alice_aligned_vec_push(vec, 2.0);
            alice_aligned_vec_push(vec, 3.0);
            assert_eq!(alice_aligned_vec_len(vec), 3);
            assert!(approx_eq(alice_aligned_vec_get(vec, 0), 1.0, EPS));
            assert!(approx_eq(alice_aligned_vec_get(vec, 2), 3.0, EPS));
            // 範囲外
            assert!(approx_eq(alice_aligned_vec_get(vec, 100), 0.0, EPS));

            alice_aligned_vec_destroy(vec);
        }
    }

    #[test]
    fn test_bitmask_operations() {
        let m = alice_bitmask_new(0);
        assert_eq!(alice_bitmask_count_ones(m), 0);

        let m = alice_bitmask_set(m, 3);
        assert_eq!(alice_bitmask_test(m, 3), 1);
        assert_eq!(alice_bitmask_test(m, 0), 0);
        assert_eq!(alice_bitmask_count_ones(m), 1);

        let m2 = alice_bitmask_set(0, 3);
        assert_eq!(alice_bitmask_and(m, m2), m);
    }

    #[test]
    fn test_branchless_ffi() {
        assert!(approx_eq(alice_select_f32(1, 10.0, 20.0), 10.0, EPS));
        assert!(approx_eq(alice_select_f32(0, 10.0, 20.0), 20.0, EPS));
        assert!(approx_eq(alice_branchless_min(3.0, 5.0), 3.0, EPS));
        assert!(approx_eq(alice_branchless_max(3.0, 5.0), 5.0, EPS));
    }

    #[test]
    fn test_fast_math_ffi() {
        assert!(approx_eq(alice_lerp(0.0, 10.0, 0.5), 5.0, EPS));
        assert!(approx_eq(alice_fma(2.0, 3.0, 4.0), 10.0, EPS));
    }

    #[test]
    fn test_null_safety() {
        unsafe {
            alice_aligned_vec_push(core::ptr::null_mut(), 1.0);
            assert_eq!(alice_aligned_vec_get(core::ptr::null(), 0), 0.0);
            assert_eq!(alice_aligned_vec_len(core::ptr::null()), 0);
            alice_aligned_vec_destroy(core::ptr::null_mut());
        }
    }

    #[test]
    fn test_bitmask_out_of_range() {
        let m = alice_bitmask_set(0, 64); // 範囲外 → 変更なし
        assert_eq!(m, 0);
        assert_eq!(alice_bitmask_test(0, 64), 0);
    }
}
