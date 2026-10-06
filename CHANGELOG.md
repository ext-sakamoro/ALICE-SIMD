# Changelog

All notable changes to ALICE-SIMD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `libm` feature (optional `libm` dependency): the float routines a `no_std` build needs (`fmaf`, `sqrtf`). A build without `std` must enable it; with neither feature the crate stops with a `compile_error!` naming both
- Tests comparing `libm::sqrtf` / `libm::fmaf` with `f32::sqrt` / `f32::mul_add` bit for bit, and the crate's `fma` / `fast_inv_sqrt` with std in both builds
- CI: tests on aarch64 and x86_64 (Linux, macOS, Windows), `thumbv7em-none-eabihf` and `wasm32-unknown-unknown` builds, clippy pedantic + nursery, MSRV, rustdoc, benches; `security-audit.yml` (cargo audit, cargo-deny, cargo-machete, stub guard) and `deny.toml`
- `scripts/preflight.sh` (the CI commands, run locally) and `scripts/docs_lint.py`
- `rust-version = "1.87"`
- `README_JP.md`

### Changed

- clippy `pedantic` / `nursery` are enabled in `Cargo.toml` `[lints.clippy]`
- `alice_bitmask_new`, `alice_bitmask_count_ones`, `alice_bitmask_and` and `alice_aligned_vec_len` are `const fn` (C ABI and behaviour unchanged)

### Fixed

- `--no-default-features` did not compile: `fast_math` called `f32::mul_add` and `f32::sqrt`, which need `std`. Without `std` they now use `libm::fmaf` / `libm::sqrtf`, which return the same bits
- README described `fast_inv_sqrt` as the Quake III approximation on every target; since 1.0.1 only `x86_64` with SSE approximates (`rsqrtss`)

## [1.0.1] - 2026-07-15

### Changed

- `fast_inv_sqrt` on targets other than `x86_64` + SSE computes `1.0 / x.sqrt()` (full precision) instead of the Quake III magic-number approximation with one Newton step, which measured 47% slower on an aarch64 core

### Added

- `benches/fast_math.rs` (criterion): `fast_rcp` / `fast_inv_sqrt` against direct division / square root and explicit NEON estimates

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
