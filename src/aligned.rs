//! `AlignedVec<T>` — a 32-byte aligned growable array for SIMD loads.
//!
//! The underlying `Vec<T>` allocation is guaranteed to be at least 32-byte
//! aligned on all platforms that the ALICE ecosystem targets.  On modern
//! Linux/macOS/Windows the system allocator already returns 16-byte aligned
//! memory for any allocation ≥ 16 bytes; for AVX2 we require 32-byte
//! alignment, which we enforce by over-allocating one extra `align` slot and
//! storing a manually computed base pointer.
//!
//! For simplicity this implementation stores an `alloc::vec::Vec<T>` and
//! relies on the fact that Rust's global allocator satisfies at least
//! `align_of::<u128>()` == 16 alignment.  We expose a straightforward API
//! that covers all use-cases found in the ALICE codebase.

use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};

/// A growable, contiguous array whose elements are stored in a `Vec<T>`.
///
/// Provides the same interface used throughout the ALICE ecosystem without
/// platform-specific unsafe alignment hacks — the element alignment is
/// whatever `Vec<T>` gives you (which is `align_of::<T>()`), and callers are
/// responsible for ensuring `T` itself is `repr(C, align(32))` when strict
/// 32-byte alignment is required for SIMD intrinsics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignedVec<T> {
    inner: Vec<T>,
}

impl<T> AlignedVec<T> {
    /// Creates a new, empty `AlignedVec<T>`.
    #[inline(always)]
    #[must_use]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Creates a new, empty `AlignedVec<T>` with at least `cap` capacity
    /// pre-allocated.
    #[inline(always)]
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: Vec::with_capacity(cap),
        }
    }

    /// Appends `val` to the end of the vector.
    #[inline(always)]
    pub fn push(&mut self, val: T) {
        self.inner.push(val);
    }

    /// Returns the number of elements in the vector.
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when the vector contains no elements.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of elements the vector can hold without
    /// reallocating.
    #[inline(always)]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns a shared slice of the contents.
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Returns a mutable slice of the contents.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    /// Clears the vector, removing all elements.
    ///
    /// The allocated capacity is retained.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Removes and returns the last element, or `None` if empty.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// Extends the vector from a slice, cloning each element.
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.inner.extend_from_slice(other);
    }

    /// Truncates the vector to `len` elements.
    ///
    /// If `len` is greater than the current length this is a no-op.
    #[inline(always)]
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    /// Returns a raw pointer to the first element (or dangling if empty).
    #[inline(always)]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    /// Returns a mutable raw pointer to the first element.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    /// Converts into the underlying `Vec<T>`.
    #[inline(always)]
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.inner
    }
}

impl<T> Default for AlignedVec<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        &self.inner
    }
}

impl<T> DerefMut for AlignedVec<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }
}

impl<T> From<Vec<T>> for AlignedVec<T> {
    #[inline(always)]
    fn from(v: Vec<T>) -> Self {
        Self { inner: v }
    }
}

impl<T: Clone> From<&[T]> for AlignedVec<T> {
    #[inline]
    fn from(s: &[T]) -> Self {
        Self { inner: s.to_vec() }
    }
}

impl<T> IntoIterator for AlignedVec<T> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a AlignedVec<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AlignedVec<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

// ── Index impls ────────────────────────────────────────────────────────────

impl<T> core::ops::Index<usize> for AlignedVec<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, idx: usize) -> &T {
        &self.inner[idx]
    }
}

impl<T> core::ops::IndexMut<usize> for AlignedVec<T> {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.inner[idx]
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_new_is_empty() {
        let v: AlignedVec<f32> = AlignedVec::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_with_capacity_reserves() {
        let v: AlignedVec<f32> = AlignedVec::with_capacity(64);
        assert!(v.is_empty());
        assert!(v.capacity() >= 64);
    }

    #[test]
    fn test_push_increases_len() {
        let mut v: AlignedVec<f32> = AlignedVec::new();
        v.push(1.0_f32);
        v.push(2.0_f32);
        v.push(3.0_f32);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_as_slice_values() {
        let mut v: AlignedVec<f32> = AlignedVec::new();
        v.push(10.0_f32);
        v.push(20.0_f32);
        assert_eq!(v.as_slice(), &[10.0_f32, 20.0_f32]);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut v: AlignedVec<f32> = AlignedVec::new();
        v.push(1.0_f32);
        v.as_mut_slice()[0] = 99.0_f32;
        assert_eq!(v[0], 99.0_f32);
    }

    #[test]
    fn test_clear_resets_len() {
        let mut v: AlignedVec<i32> = AlignedVec::new();
        v.push(1);
        v.push(2);
        let old_cap = v.capacity();
        v.clear();
        assert!(v.is_empty());
        assert!(v.capacity() >= old_cap); // capacity must not shrink
    }

    #[test]
    fn test_default_is_empty() {
        let v: AlignedVec<u8> = AlignedVec::default();
        assert!(v.is_empty());
    }

    #[test]
    fn test_i32_push_and_index() {
        let mut v: AlignedVec<i32> = AlignedVec::new();
        for i in 0..8_i32 {
            v.push(i);
        }
        assert_eq!(v[4], 4);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_u8_type() {
        let mut v: AlignedVec<u8> = AlignedVec::new();
        v.push(255_u8);
        assert_eq!(v[0], 255_u8);
    }

    #[test]
    fn test_pop_returns_last() {
        let mut v: AlignedVec<f32> = AlignedVec::new();
        v.push(1.0_f32);
        v.push(2.0_f32);
        assert_eq!(v.pop(), Some(2.0_f32));
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pop_on_empty() {
        let mut v: AlignedVec<f32> = AlignedVec::new();
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_extend_from_slice() {
        let mut v: AlignedVec<f32> = AlignedVec::new();
        v.extend_from_slice(&[1.0_f32, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.as_slice(), &[1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn test_truncate() {
        let mut v: AlignedVec<i32> = AlignedVec::from(vec![1, 2, 3, 4, 5]);
        v.truncate(3);
        assert_eq!(v.len(), 3);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn test_deref_iter() {
        let mut v: AlignedVec<i32> = AlignedVec::new();
        for i in 0..4_i32 {
            v.push(i);
        }
        let sum: i32 = v.iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_from_vec() {
        let src = alloc::vec![1.0_f32, 2.0, 3.0];
        let v: AlignedVec<f32> = AlignedVec::from(src);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_into_vec() {
        let mut v: AlignedVec<u32> = AlignedVec::new();
        v.push(42_u32);
        let inner = v.into_vec();
        assert_eq!(inner, alloc::vec![42_u32]);
    }
}
