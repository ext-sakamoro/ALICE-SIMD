# Changelog

All notable changes to ALICE-SIMD will be documented in this file.

## [1.0.0] - 2026-02-23

### Added
- `aligned` — `AlignedVec<T>` 32-byte aligned vector for SIMD loads
- `bitmask` — `BitMask64`, `ComparisonMask`, `SetBitIterator` 64-bit packed bitmask with branchless ops
- `branchless` — `select_f32`, `branchless_min`, `branchless_max`, `branchless_clamp`, `branchless_abs`
- `fast_math` — `fast_rcp`, `fast_inv_sqrt`, `fast_sqrt`, `fma`, `fma_chain`, `lerp`, `normalize`, `batch_fma`, `batch_mul_scalar`, `deg_to_rad`, `distance_squared`, `length_squared`, `Reciprocals`
- `hash` — `fnv1a` deterministic hash for content deduplication
- `SIMD_WIDTH` compile-time platform detection (AVX2=8, NEON/SSE=4)
- `align_up` SIMD-width alignment helper
- `no_std` + `alloc` core, optional `std` feature
- 110 unit tests
