//! `AlignedVec<T>` — a 32-byte aligned growable array for SIMD loads.
//!
//! The backing allocation's base pointer (`as_ptr()` / `as_mut_ptr()`) is
//! guaranteed to be at least 32-byte aligned on every platform and allocator,
//! regardless of `align_of::<T>()`. This is enough alignment for AVX2 aligned
//! loads/stores (`_mm256_load_ps` and friends) even when `T` itself does not
//! demand it (e.g. `f32`, whose natural alignment is only 4 bytes).
//!
//! The guarantee holds because `AlignedVec` does not wrap `alloc::vec::Vec<T>`
//! (whose allocation is only ever `align_of::<T>()`-aligned, with no
//! guarantee of anything stronger): it allocates directly against the global
//! allocator with an explicit [`Layout`] whose alignment is
//! `max(32, align_of::<T>())`. Even an empty, never-grown `AlignedVec` upholds
//! this: its placeholder (dangling, unallocated) pointer is constructed at
//! address `max(32, align_of::<T>())`, so the invariant holds before the
//! first allocation too.
//!
//! This replaces an earlier implementation that stored a plain `Vec<T>` and
//! relied on the observation that every allocator the ALICE ecosystem ships
//! on happens to return at least 16-byte aligned memory — which is not the
//! same guarantee as 32-byte alignment, and was never guaranteed by any
//! allocator's contract in the first place. A Windows CI run exposed the gap:
//! the allocator there returned 16-byte (not 32-byte) aligned memory for an
//! `AlignedVec<f32>`, and the crate's own doctest assertion failed.

use alloc::alloc::{alloc, dealloc, handle_alloc_error, realloc};
use alloc::vec::Vec;
use core::alloc::Layout;
use core::mem;
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};

/// Minimum base-pointer alignment, in bytes, for every `AlignedVec<T>`
/// allocation — enough for AVX2 (`_mm256_load_ps` et al.) regardless of
/// `align_of::<T>()`.
const MIN_ALIGN: usize = 32;

#[cold]
#[inline(never)]
fn capacity_overflow() -> ! {
    panic!("capacity overflow")
}

/// A growable, contiguous array that is always 32-byte aligned.
///
/// Suitable for aligned SIMD loads/stores (e.g. `_mm256_load_ps`) regardless
/// of `size_of::<T>()` / `align_of::<T>()`.
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

// SAFETY: `AlignedVec<T>` owns its buffer exclusively, the same way `Vec<T>`
// does (the `NonNull<T>` is a private implementation detail only ever used to
// access the `len` elements this `AlignedVec` itself owns), so it is
// `Send`/`Sync` under exactly the same conditions as `Vec<T>` is.
unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

impl<T> AlignedVec<T> {
    /// Whether `T` is a zero-sized type — `AlignedVec<T>` never allocates for
    /// one (there is nothing to store), matching `Vec<T>`'s own behavior.
    const IS_ZST: bool = mem::size_of::<T>() == 0;

    /// The alignment every allocation (and the dangling placeholder pointer)
    /// uses: at least `MIN_ALIGN`, but never less than `T` itself demands.
    #[inline(always)]
    const fn effective_align() -> usize {
        let t_align = mem::align_of::<T>();
        if t_align > MIN_ALIGN {
            t_align
        } else {
            MIN_ALIGN
        }
    }

    /// A non-null, correctly (over-)aligned pointer that does not point at an
    /// actual allocation — used for a `AlignedVec` that has never grown (ZST
    /// or empty), the same role `NonNull::dangling()` plays inside `Vec<T>`.
    #[inline(always)]
    const fn dangling() -> NonNull<T> {
        // `without_provenance_mut` (rather than an integer-to-pointer `as`
        // cast) builds this placeholder address as a strict-provenance
        // pointer with no provenance at all, which is the correct thing for
        // a pointer that is never actually dereferenced -- it also means
        // Miri does not need to treat this as an exposed-provenance cast.
        //
        // SAFETY: `effective_align()` is `T`'s alignment or `MIN_ALIGN`,
        // whichever is larger; both are powers of two, hence non-zero, so
        // this is a valid (non-null) pointer value to build a `NonNull` from.
        unsafe {
            NonNull::new_unchecked(core::ptr::without_provenance_mut(Self::effective_align()))
        }
    }

    /// The `Layout` an allocation of `cap` elements of `T` was (or will be)
    /// made with. Recomputed from `cap` rather than cached, so the value
    /// handed to `dealloc`/`realloc` is always exactly what the corresponding
    /// `alloc`/`realloc` call used.
    fn layout_for(cap: usize) -> Layout {
        let size = mem::size_of::<T>()
            .checked_mul(cap)
            .unwrap_or_else(|| capacity_overflow());
        Layout::from_size_align(size, Self::effective_align())
            .unwrap_or_else(|_| capacity_overflow())
    }

    /// Allocates fresh memory for `cap` elements (`cap > 0`, `T` not a ZST).
    fn alloc_new(cap: usize) -> NonNull<T> {
        let layout = Self::layout_for(cap);
        // SAFETY: `layout.size() > 0` because `cap > 0` and `T` is not a ZST
        // (the only two callers, `with_capacity` and `set_capacity`, both
        // check this before calling).
        let raw = unsafe { alloc(layout) };
        NonNull::new(raw.cast::<T>()).unwrap_or_else(|| handle_alloc_error(layout))
    }

    /// Grows (or makes, from nothing) the backing allocation to hold `new_cap`
    /// elements, preserving the already-written `[0, self.len)` prefix.
    fn set_capacity(&mut self, new_cap: usize) {
        debug_assert!(!Self::IS_ZST, "set_capacity is never called for a ZST");
        debug_assert!(
            new_cap >= self.len,
            "set_capacity must not drop live elements"
        );
        self.ptr = if self.cap == 0 {
            Self::alloc_new(new_cap)
        } else {
            let old_layout = Self::layout_for(self.cap);
            let new_layout = Self::layout_for(new_cap);
            // SAFETY: `self.ptr` was allocated by `alloc_new`/`set_capacity`
            // with exactly `old_layout` (recomputed identically from
            // `self.cap`, which is only ever changed together with the
            // allocation it describes), and `new_layout.size() > 0`.
            let raw = unsafe { realloc(self.ptr.as_ptr().cast(), old_layout, new_layout.size()) };
            NonNull::new(raw.cast::<T>()).unwrap_or_else(|| handle_alloc_error(new_layout))
        };
        self.cap = new_cap;
    }

    /// Doubles the capacity (from a baseline of 4), for `push`'s amortized
    /// growth.
    fn grow(&mut self) {
        let new_cap = if self.cap == 0 {
            4
        } else {
            self.cap
                .checked_mul(2)
                .unwrap_or_else(|| capacity_overflow())
        };
        self.set_capacity(new_cap);
    }

    /// Ensures room for `additional` more elements beyond `self.len`,
    /// growing at least to the doubling schedule `grow` would use.
    fn reserve(&mut self, additional: usize) {
        if Self::IS_ZST {
            return;
        }
        let required = self
            .len
            .checked_add(additional)
            .unwrap_or_else(|| capacity_overflow());
        if required > self.cap {
            let new_cap = required.max(self.cap.saturating_mul(2)).max(4);
            self.set_capacity(new_cap);
        }
    }

    /// Creates a new, empty `AlignedVec<T>`.
    ///
    /// No allocation is performed; `as_ptr()` returns a dangling but
    /// correctly (32-byte-or-more) aligned placeholder until the first
    /// element is pushed.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ptr: Self::dangling(),
            len: 0,
            cap: if Self::IS_ZST { usize::MAX } else { 0 },
        }
    }

    /// Creates a new, empty `AlignedVec<T>` with at least `cap` capacity
    /// pre-allocated.
    #[inline(always)]
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        if Self::IS_ZST || cap == 0 {
            return Self::new();
        }
        Self {
            ptr: Self::alloc_new(cap),
            len: 0,
            cap,
        }
    }

    /// Appends `val` to the end of the vector.
    ///
    /// # Panics
    ///
    /// Panics on capacity overflow (more elements than `isize::MAX` bytes
    /// worth of `T`), the same condition under which `Vec::push` panics.
    #[inline(always)]
    pub fn push(&mut self, val: T) {
        if !Self::IS_ZST && self.len == self.cap {
            self.grow();
        }
        // SAFETY: for a non-ZST, `self.len < self.cap` now holds (just
        // ensured by `grow()` above), so offset `self.len` is a writable,
        // not-yet-initialized slot inside the allocation. For a ZST, writing
        // at any offset of a `NonNull<T>` writes zero bytes and is always
        // sound.
        unsafe { ptr::write(self.ptr.as_ptr().add(self.len), val) };
        self.len += 1;
    }

    /// Returns the number of elements in the vector.
    #[inline(always)]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the vector contains no elements.
    #[inline(always)]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements the vector can hold without
    /// reallocating.
    #[inline(always)]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        if Self::IS_ZST {
            usize::MAX
        } else {
            self.cap
        }
    }

    /// Returns a shared slice of the contents.
    #[inline(always)]
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: `self.ptr` is valid for reads of exactly `self.len`
        // initialized `T`s (that is the class invariant maintained by every
        // method of this type), and the returned slice borrows `self`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable slice of the contents.
    #[inline(always)]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: same as `as_slice`, with exclusive access via `&mut self`.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Clears the vector, removing all elements.
    ///
    /// The allocated capacity is retained.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Removes and returns the last element, or `None` if empty.
    #[inline(always)]
    pub const fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: index `self.len` (after the decrement above) was the last
        // initialized element and is now excluded from the live
        // `[0, self.len)` range that every other method (and `Drop`) treats
        // as initialized, so reading it out here does not double-drop it.
        Some(unsafe { ptr::read(self.ptr.as_ptr().add(self.len)) })
    }

    /// Extends the vector from a slice, cloning each element.
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.reserve(other.len());
        for item in other {
            self.push(item.clone());
        }
    }

    /// Truncates the vector to `len` elements.
    ///
    /// If `len` is greater than the current length this is a no-op.
    #[inline(always)]
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        let remaining = self.len - len;
        // `self.len` is lowered *before* dropping the tail: if a `T::drop`
        // panics partway through, `AlignedVec`'s own `Drop` (which runs next,
        // during unwinding) must not also try to drop the same elements.
        self.len = len;
        // SAFETY: elements `[len, len + remaining)` (i.e. the old
        // `[len, old self.len)`) are exactly the ones being removed, and
        // `self.len` has already been lowered so nothing else observes them
        // as live.
        unsafe {
            let tail = ptr::slice_from_raw_parts_mut(self.ptr.as_ptr().add(len), remaining);
            ptr::drop_in_place(tail);
        }
    }

    /// Returns a raw pointer to the first element (or a dangling, but
    /// correctly over-aligned, placeholder if empty and never grown).
    #[inline(always)]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the first element.
    #[inline(always)]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Converts into the underlying `Vec<T>`.
    ///
    /// This always allocates a new, ordinary (`align_of::<T>()`-aligned)
    /// `Vec` and copies the elements into it, then frees `AlignedVec`'s own
    /// over-aligned buffer — a `Vec` cannot simply take ownership of this
    /// type's allocation, because `Vec`'s allocator-aware `Drop` always
    /// deallocates with the `Layout` it itself would have used
    /// (`align_of::<T>()`), which would not match an over-aligned buffer's
    /// actual layout and is immediate undefined behavior.
    #[inline]
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        let this = mem::ManuallyDrop::new(self);
        let len = this.len;
        let mut out = Vec::with_capacity(len);
        // SAFETY: `this.ptr` is valid for reads of `len` initialized `T`s,
        // `out`'s buffer has room for `len` (just reserved) and does not
        // overlap `this.ptr` (distinct allocations), and ownership of the
        // elements is being transferred bitwise to `out` — `this` is wrapped
        // in `ManuallyDrop` so its own `Drop` (which would otherwise also
        // drop these same elements) never runs.
        unsafe {
            ptr::copy_nonoverlapping(this.ptr.as_ptr(), out.as_mut_ptr(), len);
            out.set_len(len);
        }
        if !Self::IS_ZST && this.cap > 0 {
            let layout = Self::layout_for(this.cap);
            // SAFETY: `this.ptr` was allocated with exactly this `Layout`;
            // the elements it held have already been moved out above (by
            // value, not by reference), so freeing the raw memory here does
            // not drop anything a second time.
            unsafe { dealloc(this.ptr.as_ptr().cast(), layout) };
        }
        out
    }
}

impl<T> Default for AlignedVec<T> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        // SAFETY: `[0, self.len)` are exactly the initialized elements, and
        // `drop` runs at most once (ordinary `Drop` semantics), so this does
        // not double-drop.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));
        }
        if !Self::IS_ZST && self.cap > 0 {
            let layout = Self::layout_for(self.cap);
            // SAFETY: `self.ptr` was allocated by `alloc_new`/`set_capacity`
            // with exactly this `Layout` (recomputed identically from
            // `self.cap`).
            unsafe { dealloc(self.ptr.as_ptr().cast(), layout) };
        }
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut out = Self::with_capacity(self.len);
        out.extend_from_slice(self.as_slice());
        out
    }
}

impl<T: PartialEq> PartialEq for AlignedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for AlignedVec<T> {}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for AlignedVec<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> From<Vec<T>> for AlignedVec<T> {
    /// Allocates a fresh 32-byte-aligned buffer and copies `v`'s elements
    /// into it — `v`'s own buffer is only ever `align_of::<T>()`-aligned
    /// (ordinary `Vec` growth), so it cannot simply be taken over.
    fn from(mut v: Vec<T>) -> Self {
        let len = v.len();
        let mut this = Self::with_capacity(len);
        // SAFETY: `this` was just allocated with capacity `len`, and `v`'s
        // buffer holds exactly `len` valid `T`s in a distinct allocation.
        unsafe {
            ptr::copy_nonoverlapping(v.as_ptr(), this.ptr.as_ptr(), len);
            // The elements were just moved out of `v` bitwise (into `this`);
            // telling `v` its length is now 0 prevents it from dropping them
            // a second time, while its `Drop` still runs normally afterward
            // and frees its (now-empty, but still allocated) buffer.
            v.set_len(0);
        }
        this.len = len;
        this
    }
}

impl<T: Clone> From<&[T]> for AlignedVec<T> {
    #[inline]
    fn from(s: &[T]) -> Self {
        let mut v = Self::with_capacity(s.len());
        v.extend_from_slice(s);
        v
    }
}

impl<T> IntoIterator for AlignedVec<T> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a AlignedVec<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AlignedVec<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

// ── Index impls ────────────────────────────────────────────────────────────

impl<T> core::ops::Index<usize> for AlignedVec<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, idx: usize) -> &T {
        &self.as_slice()[idx]
    }
}

impl<T> core::ops::IndexMut<usize> for AlignedVec<T> {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.as_mut_slice()[idx]
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp)]
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

    // ── 32-byte alignment oracle ────────────────────────────────────────────
    //
    // The bug this guards against: `AlignedVec<T>` used to wrap a plain
    // `Vec<T>` and rely on every allocator happening to return 16-byte (not
    // 32-byte) aligned memory. On Windows that assumption was false for
    // `AlignedVec<f32>`, and the crate's own doctest assertion
    // (`buf.as_ptr() as usize % 32 == 0`) failed there. These tests check the
    // same property, but at every element count across several `T` (not just
    // `f32`) and after *every* push (not just at a single end state), so a
    // regression that only shows up after a particular growth/realloc step
    // cannot slip through.

    /// A non-power-of-two-sized, minimally aligned type, to make sure the
    /// alignment guarantee does not depend on `T`'s own size or alignment.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    struct Odd3 {
        a: u8,
        b: u8,
        c: u8,
    }

    /// A type whose own required alignment (64) is *larger* than
    /// `MIN_ALIGN` (32) — exercises the `effective_align() = align_of::<T>()`
    /// branch, not just the `effective_align() = MIN_ALIGN` branch.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(align(64))]
    struct Over64(u64);

    /// Pushes `max_len` clones of `sample` one at a time, asserting the
    /// 32-byte alignment invariant before the first push and after every
    /// single push (not only at a handful of checkpoints), so a misalignment
    /// introduced by any one growth/realloc step is caught exactly there.
    fn assert_aligned_through_growth<T: Clone + core::fmt::Debug>(sample: T, max_len: usize) {
        let mut v: AlignedVec<T> = AlignedVec::new();
        assert_eq!(
            v.as_ptr() as usize % 32,
            0,
            "AlignedVec<{}>::new() (pre-push, dangling) pointer is not 32-byte aligned",
            core::any::type_name::<T>(),
        );
        for i in 0..max_len {
            v.push(sample.clone());
            assert_eq!(
                v.as_ptr() as usize % 32,
                0,
                "AlignedVec<{}> misaligned after push #{} (len={}, cap={})",
                core::any::type_name::<T>(),
                i + 1,
                v.len(),
                v.capacity(),
            );
        }
    }

    #[test]
    fn test_alignment_oracle_u8_every_push() {
        assert_aligned_through_growth::<u8>(7, 1000);
    }

    #[test]
    fn test_alignment_oracle_f32_every_push() {
        // The exact type and scenario (`AlignedVec<f32>`) the Windows CI
        // failure and the crate's doctest use.
        assert_aligned_through_growth::<f32>(1.5, 1000);
    }

    #[test]
    fn test_alignment_oracle_u64_every_push() {
        assert_aligned_through_growth::<u64>(42, 1000);
    }

    #[test]
    fn test_alignment_oracle_odd_sized_type_every_push() {
        assert_aligned_through_growth(Odd3 { a: 1, b: 2, c: 3 }, 1000);
    }

    #[test]
    fn test_alignment_oracle_over_aligned_type_every_push() {
        assert_aligned_through_growth(Over64(7), 1000);
    }

    #[test]
    fn test_alignment_zero_length_new_does_not_panic_or_allocate() {
        let v: AlignedVec<f32> = AlignedVec::new();
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        // Dangling, but intentionally constructed to already satisfy the
        // 32-byte guarantee (see `AlignedVec::dangling`), so asserting on it
        // here is sound and meaningful, not merely "didn't crash".
        assert_eq!(v.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn test_alignment_zero_capacity_with_capacity_does_not_panic_or_allocate() {
        let v: AlignedVec<u64> = AlignedVec::with_capacity(0);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        assert_eq!(v.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn test_alignment_zst_does_not_allocate_and_is_well_aligned() {
        let mut v: AlignedVec<()> = AlignedVec::new();
        assert_eq!(v.capacity(), usize::MAX);
        assert_eq!(v.as_ptr() as usize % 32, 0);
        for _ in 0..10 {
            v.push(());
        }
        assert_eq!(v.len(), 10);
        assert_eq!(v.as_ptr() as usize % 32, 0);
    }
}
