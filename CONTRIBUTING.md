# Contributing to ALICE-SIMD

## Build

```bash
cargo build
```

## Test

```bash
cargo test
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **`no_std` + `alloc`**: core library has zero external dependencies.
- **32-byte alignment**: `AlignedVec<T>` ensures AVX2-friendly memory layout.
- **Branchless primitives**: all select/min/max/clamp/abs use bit manipulation, no conditional branches.
- **Platform-adaptive `SIMD_WIDTH`**: compile-time detection (AVX2=8, NEON/SSE=4).
- **Division exorcism**: `Reciprocals` table and `fast_rcp` avoid expensive division in hot loops.
