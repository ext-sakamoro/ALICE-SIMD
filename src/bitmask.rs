//! `BitMask64` — 64-bit packed bitmask with branchless operations.
//!
//! Represents up to 64 boolean flags in a single `u64`.  All operations are
//! branchless and map directly to hardware instructions (`popcnt`, `lzcnt`,
//! `tzcnt`).

// ── BitMask64 ──────────────────────────────────────────────────────────────

/// A 64-bit packed bitmask.
///
/// Bit `i` represents the boolean state of element `i` (0-indexed from LSB).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BitMask64(pub u64);

impl BitMask64 {
    /// All 64 bits set — every element is `true`.
    pub const ALL_TRUE: Self = Self(u64::MAX);
    /// All 64 bits clear — every element is `false`.
    pub const ALL_FALSE: Self = Self(0);

    /// Creates a mask from a single `bool`: `true` → `ALL_TRUE`, `false` → `ALL_FALSE`.
    #[inline(always)]
    #[must_use]
    pub fn from_bool(b: bool) -> Self {
        // branchless: -(true as i64) == -1 == 0xFFFF…
        Self((-i64::from(b)) as u64)
    }

    /// Returns a new mask with bit `i` set.
    #[inline(always)]
    #[must_use]
    pub fn set(self, i: u32) -> Self {
        debug_assert!(i < 64, "bit index out of range");
        Self(self.0 | (1_u64 << i))
    }

    /// Returns a new mask with bit `i` cleared.
    #[inline(always)]
    #[must_use]
    pub fn clear(self, i: u32) -> Self {
        debug_assert!(i < 64, "bit index out of range");
        Self(self.0 & !(1_u64 << i))
    }

    /// Returns `true` if bit `i` is set.
    #[inline(always)]
    #[must_use]
    pub fn test(self, i: u32) -> bool {
        debug_assert!(i < 64, "bit index out of range");
        (self.0 >> i) & 1 != 0
    }

    /// Bitwise AND.
    #[inline(always)]
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Bitwise OR.
    #[inline(always)]
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Bitwise NOT (complement).
    ///
    /// Prefer the `!` operator (`core::ops::Not`) which delegates to this.
    #[inline(always)]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(!self.0)
    }

    /// Bitwise XOR.
    #[inline(always)]
    #[must_use]
    pub fn xor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// Number of bits that are set (population count / `popcnt`).
    #[inline(always)]
    #[must_use]
    pub fn count_ones(self) -> u32 {
        self.0.count_ones()
    }

    /// Number of leading zero bits.
    #[inline(always)]
    #[must_use]
    pub fn leading_zeros(self) -> u32 {
        self.0.leading_zeros()
    }

    /// Number of trailing zero bits.
    #[inline(always)]
    #[must_use]
    pub fn trailing_zeros(self) -> u32 {
        self.0.trailing_zeros()
    }

    /// Returns `true` if at least one bit is set.
    #[inline(always)]
    #[must_use]
    pub fn any(self) -> bool {
        self.0 != 0
    }

    /// Returns `true` if all 64 bits are set.
    #[inline(always)]
    #[must_use]
    pub fn all(self) -> bool {
        self.0 == u64::MAX
    }

    /// Returns `true` if no bits are set.
    #[inline(always)]
    #[must_use]
    pub fn none(self) -> bool {
        self.0 == 0
    }

    /// Blends two slices element-wise using this mask.
    ///
    /// For each index `i < 64` and `i < a.len().min(b.len())`:
    /// - Writes `a[i]` to `dst[i]` if bit `i` is **set**.
    /// - Writes `b[i]` to `dst[i]` if bit `i` is **clear**.
    ///
    /// `dst` must be at least `a.len().min(b.len())` long.
    #[inline]
    pub fn blend_slices<T: Copy>(self, a: &[T], b: &[T], dst: &mut [T]) {
        let len = a.len().min(b.len()).min(dst.len()).min(64);
        // Use enumerate to avoid needless-range-loop lint
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate().take(len) {
            let use_a = self.test(i as u32);
            dst[i] = if use_a { *av } else { *bv };
        }
    }

    /// Returns an iterator over the indices of all **set** bits.
    #[inline(always)]
    #[must_use]
    pub fn iter_set_bits(self) -> SetBitIterator {
        SetBitIterator { mask: self.0 }
    }
}

impl core::ops::BitAnd for BitMask64 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        self.and(rhs)
    }
}

impl core::ops::BitOr for BitMask64 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        self.or(rhs)
    }
}

impl core::ops::BitXor for BitMask64 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        self.xor(rhs)
    }
}

impl core::ops::Not for BitMask64 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        BitMask64::not(self)
    }
}

impl core::fmt::Binary for BitMask64 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Binary::fmt(&self.0, f)
    }
}

// ── SetBitIterator ─────────────────────────────────────────────────────────

/// An iterator over the indices of set bits in a `u64` mask.
///
/// Uses `trailing_zeros` + bit-clearing for `O(k)` where `k` = number of set
/// bits, which is optimal.
#[derive(Clone, Debug)]
pub struct SetBitIterator {
    mask: u64,
}

impl Iterator for SetBitIterator {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.mask == 0 {
            return None;
        }
        let idx = self.mask.trailing_zeros() as usize;
        // Clear the lowest set bit: mask &= mask - 1
        self.mask &= self.mask - 1;
        Some(idx)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.mask.count_ones() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for SetBitIterator {}

// ── ComparisonMask ─────────────────────────────────────────────────────────

/// Builds a `BitMask64` by comparing slices element-wise.
pub struct ComparisonMask;

impl ComparisonMask {
    /// Sets bit `i` if `a[i] > b[i]` (for `f32` slices, up to 64 elements).
    #[inline]
    #[must_use]
    pub fn gt(a: &[f32], b: &[f32]) -> BitMask64 {
        let len = a.len().min(b.len()).min(64);
        let bits = a
            .iter()
            .zip(b.iter())
            .take(len)
            .enumerate()
            .fold(0_u64, |acc, (i, (av, bv))| acc | u64::from(*av > *bv) << i);
        BitMask64(bits)
    }

    /// Sets bit `i` if `a[i] == b[i]` (for `i32` slices, up to 64 elements).
    #[inline]
    #[must_use]
    pub fn eq_i32(a: &[i32], b: &[i32]) -> BitMask64 {
        let len = a.len().min(b.len()).min(64);
        let bits = a
            .iter()
            .zip(b.iter())
            .take(len)
            .enumerate()
            .fold(0_u64, |acc, (i, (av, bv))| acc | u64::from(*av == *bv) << i);
        BitMask64(bits)
    }

    /// Sets bit `i` if `a[i] != 0` (for `i32` slices, up to 64 elements).
    #[inline]
    #[must_use]
    pub fn nonzero(a: &[i32]) -> BitMask64 {
        let len = a.len().min(64);
        let bits = a
            .iter()
            .take(len)
            .enumerate()
            .fold(0_u64, |acc, (i, v)| acc | u64::from(*v != 0) << i);
        BitMask64(bits)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_true_all_false() {
        assert!(BitMask64::ALL_TRUE.all());
        assert!(BitMask64::ALL_FALSE.none());
        assert!(!BitMask64::ALL_TRUE.none());
        assert!(!BitMask64::ALL_FALSE.any());
    }

    #[test]
    fn test_from_bool_true() {
        assert_eq!(BitMask64::from_bool(true), BitMask64::ALL_TRUE);
    }

    #[test]
    fn test_from_bool_false() {
        assert_eq!(BitMask64::from_bool(false), BitMask64::ALL_FALSE);
    }

    #[test]
    fn test_set_and_test() {
        let m = BitMask64::ALL_FALSE.set(7);
        assert!(m.test(7));
        assert!(!m.test(0));
        assert!(!m.test(63));
    }

    #[test]
    fn test_clear() {
        let m = BitMask64::ALL_TRUE.clear(3);
        assert!(!m.test(3));
        assert!(m.test(0));
        assert!(m.test(63));
    }

    #[test]
    fn test_and() {
        let a = BitMask64(0xF0F0);
        let b = BitMask64(0xFF00);
        assert_eq!(a.and(b), BitMask64(0xF000));
    }

    #[test]
    fn test_or() {
        let a = BitMask64(0x00FF);
        let b = BitMask64(0xFF00);
        assert_eq!(a.or(b), BitMask64(0xFFFF));
    }

    #[test]
    fn test_not() {
        assert_eq!(BitMask64::ALL_FALSE.not(), BitMask64::ALL_TRUE);
    }

    #[test]
    fn test_xor() {
        let a = BitMask64(0xAAAA);
        let b = BitMask64(0xFFFF);
        assert_eq!(a.xor(b), BitMask64(0x5555));
    }

    #[test]
    fn test_count_ones() {
        assert_eq!(BitMask64(0b1010_1010).count_ones(), 4);
        assert_eq!(BitMask64::ALL_TRUE.count_ones(), 64);
        assert_eq!(BitMask64::ALL_FALSE.count_ones(), 0);
    }

    #[test]
    fn test_leading_trailing_zeros() {
        let m = BitMask64(0b1000); // bit 3 only
        assert_eq!(m.trailing_zeros(), 3);
        assert_eq!(m.leading_zeros(), 60);
    }

    #[test]
    fn test_any_all_none() {
        let partial = BitMask64(0x1);
        assert!(partial.any());
        assert!(!partial.all());
        assert!(!partial.none());
    }

    #[test]
    fn test_blend_slices() {
        let a = [10_i32, 20, 30, 40];
        let b = [1_i32, 2, 3, 4];
        // mask: bits 0 and 2 set → select a[0], b[1], a[2], b[3]
        let mask = BitMask64(0b0101);
        let mut dst = [0_i32; 4];
        mask.blend_slices(&a, &b, &mut dst);
        assert_eq!(dst, [10, 2, 30, 4]);
    }

    #[test]
    fn test_iter_set_bits_order() {
        let m = BitMask64(0b1010_1010);
        let indices: alloc::vec::Vec<usize> = m.iter_set_bits().collect();
        assert_eq!(indices, alloc::vec![1, 3, 5, 7]);
    }

    #[test]
    fn test_iter_set_bits_empty() {
        let m = BitMask64::ALL_FALSE;
        assert_eq!(m.iter_set_bits().count(), 0);
    }

    #[test]
    fn test_iter_set_bits_all() {
        let m = BitMask64::ALL_TRUE;
        assert_eq!(m.iter_set_bits().count(), 64);
    }

    #[test]
    fn test_set_bit_iterator_exact_size() {
        let m = BitMask64(0b111);
        let it = m.iter_set_bits();
        assert_eq!(it.len(), 3);
    }

    #[test]
    fn test_comparison_mask_gt() {
        let a = [1.0_f32, 5.0, 3.0, 2.0];
        let b = [0.0_f32, 5.0, 4.0, 1.0];
        let m = ComparisonMask::gt(&a, &b);
        // a[0]>b[0]=true, a[1]>b[1]=false, a[2]>b[2]=false, a[3]>b[3]=true
        assert!(m.test(0));
        assert!(!m.test(1));
        assert!(!m.test(2));
        assert!(m.test(3));
    }

    #[test]
    fn test_comparison_mask_eq_i32() {
        let a = [1_i32, 2, 3];
        let b = [1_i32, 0, 3];
        let m = ComparisonMask::eq_i32(&a, &b);
        assert!(m.test(0));
        assert!(!m.test(1));
        assert!(m.test(2));
    }

    #[test]
    fn test_comparison_mask_nonzero() {
        let a = [0_i32, 5, 0, -1];
        let m = ComparisonMask::nonzero(&a);
        assert!(!m.test(0));
        assert!(m.test(1));
        assert!(!m.test(2));
        assert!(m.test(3));
    }

    #[test]
    fn test_operator_overloads() {
        let a = BitMask64(0xF0);
        let b = BitMask64(0x0F);
        assert_eq!((a & b), BitMask64(0x00));
        assert_eq!((a | b), BitMask64(0xFF));
        assert_eq!((a ^ b), BitMask64(0xFF));
        assert_eq!(!BitMask64::ALL_FALSE, BitMask64::ALL_TRUE);
    }
}
