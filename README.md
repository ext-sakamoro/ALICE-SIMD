# ALICE-SIMD

Shared SIMD, branchless, and fast-math primitives for the ALICE ecosystem.

**Zero external dependencies.** Pure Rust, `no_std` + `alloc` only.

## Modules

| Module | Description |
|--------|-------------|
| `aligned` | `AlignedVec<T>` — 32-byte aligned vector for SIMD loads/stores |
| `bitmask` | `BitMask64` — 64-bit packed bitmask with branchless set/test/iterate |
| `branchless` | `select_f32`, `min`, `max`, `clamp`, `abs`, `signum` without branches |
| `fast_math` | Reciprocals, inverse sqrt, FMA, lerp, exp, RMSNorm, softmax |
| `hash` | `fnv1a` — deterministic hash for content deduplication |
| `vec3` | `Vec3`, `Vec4` — compact vector math types |
| `ffi` | C-ABI exports (15 functions, feature `ffi`) |

## fast_math API

| Function | Description | Use Case |
|----------|-------------|----------|
| `fast_rcp(x)` | Newton-Raphson reciprocal | Division avoidance |
| `fast_inv_sqrt(x)` | Quake III inverse sqrt | Normalization |
| `fast_sqrt(x)` | `x * fast_inv_sqrt(x)` | Distance calculation |
| `fast_exp(x)` | Range reduction + polynomial (~0.3% error) | Softmax, activation |
| `fma(a,b,c)` | Fused multiply-add | Dot products |
| `lerp(a,b,t)` | Linear interpolation | Animation, blending |
| `rmsnorm(x, out, eps)` | RMS normalization (DPS) | LLM layer norm |
| `rmsnorm_inplace(x, eps)` | RMS normalization (in-place) | LLM layer norm |
| `softmax(x, out)` | Numerically stable softmax (DPS) | Attention, gating |
| `softmax_inplace(x)` | Numerically stable softmax (in-place) | Attention, gating |
| `batch_mul_scalar(data, s)` | Vectorized scalar multiply | Batch processing |
| `batch_fma(data, a, b)` | Vectorized FMA | Batch processing |
| `normalize(val, inv_max)` | Value normalization | Colour/coordinate mapping |
| `RECIPROCALS` | Pre-computed 1/255, 1/320, etc. | Hot-path division avoidance |

## Example

```rust
use alice_simd::{fast_exp, rmsnorm, softmax, AlignedVec, BitMask64, branchless_min};

// Fast math
let e = fast_exp(1.0); // ~2.718 (< 0.3% error)

// RMSNorm (LLM layer normalization)
let x = [1.0_f32, 2.0, 3.0, 4.0];
let mut out = [0.0_f32; 4];
rmsnorm(&x, &mut out, 1e-6);

// Softmax
let logits = [1.0_f32, 2.0, 3.0];
let mut probs = [0.0_f32; 3];
softmax(&logits, &mut probs);

// Branchless operations
let v = branchless_min(3.14, 2.72); // 2.72, no branch

// SIMD-aligned allocation
let mut buf = AlignedVec::<f32>::new();
buf.push(1.0);

// Bitmask operations
let mask = BitMask64::new(0xFF);
assert_eq!(mask.count_ones(), 8);
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `ffi` | No | C-ABI FFI exports (15 functions) |

## Platform

| Target | SIMD Width | Notes |
|--------|-----------|-------|
| x86_64 + AVX2 | 8 lanes | `f32 x 8` |
| x86_64 (SSE) | 4 lanes | `f32 x 4` |
| aarch64 (NEON) | 4 lanes | `f32 x 4`, Apple Silicon / Raspberry Pi 5 |

Apple Silicon (M1/M2/M3/M4) と Raspberry Pi 5 は同じ ARM NEON 命令セットを使用。
リコンパイルのみで両環境で動作し、`fast_exp`/`rmsnorm`/`softmax` 等の
LLM推論プリミティブを統一コードで提供する。

## Quality

| Metric | Value |
|--------|-------|
| clippy (pedantic+nursery) | 0 warnings |
| Tests | 160 |
| Dependencies | 0 (no_std + alloc) |
| fmt | clean |

## License

See [LICENSE](LICENSE).
