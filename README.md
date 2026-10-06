# ALICE-SIMD

[日本語](README_JP.md)

Shared SIMD-friendly, branchless and fast-math primitives for the ALICE
ecosystem: 32-byte aligned buffers, 64-bit masks, branch-free `f32` selection,
fused multiply-add helpers, RMSNorm / softmax and a deterministic FNV-1a hash.
The library is `no_std` + `alloc`, has no required dependencies, and exports a
C ABI for Unity and Unreal Engine.

It is a set of scalar building blocks written so that LLVM can vectorize the
loops that use them; it does not provide portable SIMD vector types or
explicit intrinsics beyond `rcpss` / `rsqrtss` on `x86_64`. For explicit vector
types use `core::arch`, `wide` or `std::simd`.

License: MIT OR Apache-2.0

## Contents

- [Installation](#installation)
- [Example](#example)
- [Features](#features)
- [Modules](#modules)
- [fast_math API](#fast_math-api)
- [Platforms and SIMD width](#platforms-and-simd-width)
- [no_std](#no_std)
- [Performance](#performance)
- [Minimum supported Rust version](#minimum-supported-rust-version)
- [Building and testing](#building-and-testing)
- [License](#license)

## Installation

The crate is not published on crates.io; depend on the repository:

```toml
[dependencies]
alice-simd = { git = "https://github.com/ext-sakamoro/ALICE-SIMD" }
```

## Example

```rust
use alice_simd::{branchless_min, fast_exp, fnv1a, rmsnorm, softmax, AlignedVec, BitMask64};

// fast exp (< 0.3% relative error)
let e = fast_exp(1.0);
assert!((e - core::f32::consts::E).abs() < 0.01);

// RMSNorm (LLM layer normalization), destination-passing style
let x = [1.0_f32, 2.0, 3.0, 4.0];
let mut out = [0.0_f32; 4];
rmsnorm(&x, &mut out, 1e-6);

// numerically stable softmax
let logits = [1.0_f32, 2.0, 3.0];
let mut probs = [0.0_f32; 3];
softmax(&logits, &mut probs);
assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-3);

// branch-free min
assert_eq!(branchless_min(3.0, 2.0), 2.0);

// 32-byte aligned buffer for SIMD loads / stores
let mut buf = AlignedVec::<f32>::new();
buf.push(1.0);
assert_eq!(buf.as_ptr() as usize % 32, 0);

// 64-bit mask
let mask = BitMask64(0xFF).set(10);
assert_eq!(mask.count_ones(), 9);
assert!(mask.test(10));

// deterministic content hash
let h = fnv1a(b"alice");
assert_eq!(h, fnv1a(b"alice"));
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Uses `f32::mul_add` / `f32::sqrt` from the standard library |
| `libm` | no | `no_std` float routines (`libm::fmaf` / `libm::sqrtf`); required when `std` is off |
| `ffi` | no | C ABI exports (15 functions, implies `std`), see `bindings/` |

`std` and `libm` produce bit-identical results: both pairs of functions are the
IEEE 754 correctly rounded `fusedMultiplyAdd` / `squareRoot`, and the test
suite compares them bit for bit.

## Modules

| Module | Contents |
|--------|----------|
| `aligned` | `AlignedVec<T>` — 32-byte aligned vector for SIMD loads / stores |
| `bitmask` | `BitMask64`, `ComparisonMask`, `SetBitIterator` — 64-bit packed mask with branch-free set / test / blend / iterate |
| `branchless` | `select_f32`, `branchless_min` / `max` / `clamp` / `abs` / `sign` / `signum`, `is_negative` |
| `fast_math` | reciprocals, inverse square root, FMA, lerp, `fast_exp`, RMSNorm, softmax, batch helpers, `RECIPROCALS` |
| `hash` | `fnv1a`, `fnv1a_u32`, `fnv1a_u64`, `fnv1a_pair` — deterministic FNV-1a |
| `vec3` | `Vec3`, `Vec4` — compact vector types (dot, cross, normalize, lerp) |
| `ffi` | C ABI (feature `ffi`): `AlignedVec<f32>` (5), `BitMask64` (5), branchless (3), fast math (2) |

Crate root: `SIMD_WIDTH` (lanes of `f32` for the compile target) and
`align_up(n)` (round up to a multiple of `SIMD_WIDTH`).

## fast_math API

| Function | Implementation | Use |
|----------|----------------|-----|
| `fast_rcp(x)` | `x86_64` + SSE: `rcpss` (~12-bit); other targets: `1.0 / x` | division avoidance |
| `fast_inv_sqrt(x)` | `x86_64` + SSE: `rsqrtss` (~12-bit); other targets: `1.0 / x.sqrt()` (correctly rounded square root) | normalization |
| `fast_sqrt(x)` | `x * fast_inv_sqrt(x)` | distances |
| `fast_exp(x)` | range reduction + degree-3 polynomial (< 0.3% relative error) | softmax, activations |
| `fma(a, b, c)` | `a * b + c`, one rounding | dot products |
| `fma_chain(a, b, c, d)` | `a * b + c * d` | bilinear terms |
| `lerp(a, b, t)` | `a + t * (b - a)` as one FMA | blending |
| `distance_squared`, `length_squared` | 2-D squared lengths via FMA | comparisons without `sqrt` |
| `rmsnorm(x, out, eps)`, `rmsnorm_inplace(x, eps)` | RMS normalization | LLM layer norm |
| `softmax(x, out)`, `softmax_inplace(x)` | max-subtracted softmax | attention, gating |
| `batch_mul_scalar`, `batch_fma` | in-place loops written for auto-vectorization | batch processing |
| `deg_to_rad`, `normalize` | multiply by a constant / precomputed reciprocal | hot-path division avoidance |
| `RECIPROCALS` | `1/255`, `1/320`, `1/360`, `1/1024`, ... | hot-path division avoidance |

`fast_rcp` / `fast_inv_sqrt` therefore return approximations on `x86_64` and
exact IEEE results elsewhere; code that needs identical results on every target
should use `1.0 / x` and `fast_math::fma` directly.

## Platforms and SIMD width

| Target | `SIMD_WIDTH` | Notes |
|--------|--------------|-------|
| `x86_64` with AVX2 enabled (`-C target-feature=+avx2`) | 8 | `f32 x 8` |
| `x86_64` (SSE2 baseline) | 4 | `f32 x 4`, `rcpss` / `rsqrtss` in `fast_rcp` / `fast_inv_sqrt` |
| `aarch64` (NEON) | 4 | `f32 x 4` |
| other (`thumbv7em`, `wasm32`, ...) | 4 | scalar paths |

CI tests aarch64 (Linux, macOS) and x86_64 (Linux, Windows), and builds
`thumbv7em-none-eabihf` (no_std) and `wasm32-unknown-unknown`.

## no_std

The library is `#![no_std]` and uses `alloc` (for `AlignedVec`). Without `std`,
enable `libm`:

```toml
[dependencies]
alice-simd = { git = "https://github.com/ext-sakamoro/ALICE-SIMD", default-features = false, features = ["libm"] }
```

A build with neither `std` nor `libm` stops with a `compile_error!`. CI builds
the library for `thumbv7em-none-eabihf`, a target without `std`, and runs the
unit tests with the `libm` code path on the host.

## Performance

<!-- perf-measured: 2026-10-07 benches/fast_math.rs -->
`cargo bench` (criterion, release), arm64. Each figure is the time for 1024
calls on values `0.1 ..= 102.4`, with criterion's 95% interval in brackets:

| Benchmark | Time |
|-----------|------|
| `fast_rcp` (`1.0 / x` on aarch64) | 3.33 µs [2.89, 3.79] |
| `1.0 / x` written directly | 2.17 µs [1.91, 2.48] |
| NEON `frecpe` + one `frecps` (~16-bit) | 2.17 µs [1.98, 2.37] |
| NEON `frecpe` only (~8-bit) | 1.72 µs [1.65, 1.80] |
| `fast_inv_sqrt` (`1.0 / x.sqrt()` on aarch64) | 2.01 µs [1.87, 2.16] |
| `1.0 / x.sqrt()` written directly | 2.13 µs [1.96, 2.32] |
| NEON `frsqrte` + one `frsqrts` (~16-bit) | 2.26 µs [2.08, 2.45] |

On aarch64, `fast_rcp` and `fast_inv_sqrt` compile to the same code as the
direct expressions, so the differences between those rows are measurement
spread. The NEON estimates with one refinement step are within that spread
and have ~16-bit precision; the bare 8-bit estimate is faster but too coarse
for normalization, so none of them is used. Figures differ by CPU; run `cargo bench` on
the target hardware.

## Minimum supported Rust version

Rust 1.87 (`rust-version` in `Cargo.toml`), checked in CI for the library with
default and all features. 1.86 rejects the `const fn` wrappers over
`Vec::len` / `Vec::as_slice` in `AlignedVec`. Development and CI use the
toolchain pinned in `rust-toolchain.toml`.

## Building and testing

```sh
cargo test --all-features                                     # unit + doc tests
cargo test --lib --no-default-features --features libm        # no_std code path on the host
cargo build --lib --no-default-features --features libm --target thumbv7em-none-eabihf
cargo clippy --all-targets --all-features -- -D warnings      # pedantic + nursery (Cargo.toml)
cargo bench
scripts/preflight.sh            # every CI step with the same arguments
scripts/preflight.sh --quick    # static checks, builds, docs and `cargo test --lib`
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE](LICENSE))

at your option.
