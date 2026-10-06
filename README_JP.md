# ALICE-SIMD

[English](README.md)

ALICE エコシステム共通の SIMD 向け・分岐なし・高速数学プリミティブ集 32 byte
境界に揃えたバッファ、64 bit マスク、分岐なしの `f32` 選択、融合積和 (FMA) の
ヘルパー、RMSNorm / softmax、決定的な FNV-1a ハッシュを含む ライブラリは
`no_std` + `alloc` で必須の依存は無く、Unity と Unreal Engine 向けに C ABI を
公開する

LLVM が利用側のループを自動ベクトル化できる形で書いたスカラーの部品集であり、
可搬な SIMD ベクトル型や、`x86_64` の `rcpss` / `rsqrtss` 以外の明示的な
intrinsics は提供しない 明示的なベクトル型が必要なら `core::arch`、`wide`、
`std::simd` を使う

License: MIT OR Apache-2.0

## 目次

- [インストール](#インストール)
- [使用例](#使用例)
- [Feature](#feature)
- [モジュール](#モジュール)
- [fast_math API](#fast_math-api)
- [プラットフォームと SIMD 幅](#プラットフォームと-simd-幅)
- [no_std](#no_std)
- [性能](#性能)
- [最小サポート Rust バージョン](#最小サポート-rust-バージョン)
- [ビルドとテスト](#ビルドとテスト)
- [ライセンス](#ライセンス)

## インストール

crates.io には公開していない リポジトリを直接指定する

```toml
[dependencies]
alice-simd = { git = "https://github.com/ext-sakamoro/ALICE-SIMD" }
```

## 使用例

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

## Feature

| Feature | 既定 | 内容 |
|---------|------|------|
| `std` | 有効 | 標準ライブラリの `f32::mul_add` / `f32::sqrt` を使う |
| `libm` | 無効 | `no_std` 用の浮動小数点関数 (`libm::fmaf` / `libm::sqrtf`)、`std` を外す場合は必須 |
| `ffi` | 無効 | C ABI (15 関数、`std` を含む)、`bindings/` を参照 |

`std` と `libm` は bit 単位で同じ結果を返す どちらも IEEE 754 の正しく丸めた
`fusedMultiplyAdd` / `squareRoot` であり、テストで bit 単位に比較している

## モジュール

| モジュール | 内容 |
|-----------|------|
| `aligned` | `AlignedVec<T>` — SIMD のロード / ストア向けに 32 byte 境界に揃えたベクタ |
| `bitmask` | `BitMask64`、`ComparisonMask`、`SetBitIterator` — 分岐なしの set / test / blend / 反復ができる 64 bit マスク |
| `branchless` | `select_f32`、`branchless_min` / `max` / `clamp` / `abs` / `sign` / `signum`、`is_negative` |
| `fast_math` | 逆数、逆平方根、FMA、lerp、`fast_exp`、RMSNorm、softmax、バッチ処理、`RECIPROCALS` |
| `hash` | `fnv1a`、`fnv1a_u32`、`fnv1a_u64`、`fnv1a_pair` — 決定的な FNV-1a |
| `vec3` | `Vec3`、`Vec4` — 小さなベクトル型 (dot、cross、normalize、lerp) |
| `ffi` | C ABI (feature `ffi`): `AlignedVec<f32>` (5)、`BitMask64` (5)、branchless (3)、高速数学 (2) |

crate 直下: `SIMD_WIDTH` (コンパイル対象での `f32` レーン数) と `align_up(n)`
(`SIMD_WIDTH` の倍数への切り上げ)

## fast_math API

| 関数 | 実装 | 用途 |
|------|------|------|
| `fast_rcp(x)` | `x86_64` + SSE: `rcpss` (約 12 bit)、その他: `1.0 / x` | 除算の回避 |
| `fast_inv_sqrt(x)` | `x86_64` + SSE: `rsqrtss` (約 12 bit)、その他: `1.0 / x.sqrt()` (正しく丸めた平方根) | 正規化 |
| `fast_sqrt(x)` | `x * fast_inv_sqrt(x)` | 距離 |
| `fast_exp(x)` | 範囲縮小 + 3 次多項式 (相対誤差 0.3% 未満) | softmax、活性化関数 |
| `fma(a, b, c)` | `a * b + c` を 1 回の丸めで | 内積 |
| `fma_chain(a, b, c, d)` | `a * b + c * d` | 双線形の項 |
| `lerp(a, b, t)` | `a + t * (b - a)` を FMA 1 回で | ブレンド |
| `distance_squared`、`length_squared` | FMA による 2 次元の距離の 2 乗 | `sqrt` なしの比較 |
| `rmsnorm(x, out, eps)`、`rmsnorm_inplace(x, eps)` | RMS 正規化 | LLM の層正規化 |
| `softmax(x, out)`、`softmax_inplace(x)` | 最大値を引いた softmax | attention、gating |
| `batch_mul_scalar`、`batch_fma` | 自動ベクトル化向けに書いた in-place ループ | バッチ処理 |
| `deg_to_rad`、`normalize` | 定数 / 事前計算した逆数との乗算 | ホットパスの除算回避 |
| `RECIPROCALS` | `1/255`、`1/320`、`1/360`、`1/1024` など | ホットパスの除算回避 |

`fast_rcp` / `fast_inv_sqrt` は `x86_64` では近似値、それ以外では IEEE の正確な
値を返す 全ターゲットで同じ結果が必要なら `1.0 / x` と `fast_math::fma` を
直接使う

## プラットフォームと SIMD 幅

| ターゲット | `SIMD_WIDTH` | 備考 |
|-----------|--------------|------|
| `x86_64` で AVX2 を有効化 (`-C target-feature=+avx2`) | 8 | `f32 x 8` |
| `x86_64` (SSE2 ベースライン) | 4 | `f32 x 4`、`fast_rcp` / `fast_inv_sqrt` で `rcpss` / `rsqrtss` |
| `aarch64` (NEON) | 4 | `f32 x 4` |
| その他 (`thumbv7em`、`wasm32` など) | 4 | スカラー経路 |

CI は aarch64 (Linux、macOS) と x86_64 (Linux、Windows) でテストし、
`thumbv7em-none-eabihf` (no_std) と `wasm32-unknown-unknown` をビルドする

## no_std

ライブラリは `#![no_std]` で、`AlignedVec` のために `alloc` を使う `std` を
外す場合は `libm` を有効にする

```toml
[dependencies]
alice-simd = { git = "https://github.com/ext-sakamoro/ALICE-SIMD", default-features = false, features = ["libm"] }
```

`std` と `libm` のどちらも無い構成は `compile_error!` で止まる CI は `std` を
持たないターゲット `thumbv7em-none-eabihf` 向けにライブラリをビルドし、さらに
`libm` 経路の単体テストをホスト上で実行する

## 性能

<!-- perf-measured: 2026-10-07 benches/fast_math.rs -->
`cargo bench` (criterion、release)、arm64 各値は `0.1 ..= 102.4` の値に対する
1024 回の呼び出し時間で、括弧内は criterion の 95% 区間

| ベンチマーク | 時間 |
|-------------|------|
| `fast_rcp` (aarch64 では `1.0 / x`) | 3.33 µs [2.89, 3.79] |
| `1.0 / x` を直接記述 | 2.17 µs [1.91, 2.48] |
| NEON `frecpe` + `frecps` 1 回 (約 16 bit) | 2.17 µs [1.98, 2.37] |
| NEON `frecpe` のみ (約 8 bit) | 1.72 µs [1.65, 1.80] |
| `fast_inv_sqrt` (aarch64 では `1.0 / x.sqrt()`) | 2.01 µs [1.87, 2.16] |
| `1.0 / x.sqrt()` を直接記述 | 2.13 µs [1.96, 2.32] |
| NEON `frsqrte` + `frsqrts` 1 回 (約 16 bit) | 2.26 µs [2.08, 2.45] |

aarch64 では `fast_rcp` / `fast_inv_sqrt` は直接記述した式と同じコードになる
ので、それらの行の差は測定のばらつき 1 回の補正付きの NEON 推定値はばらつきの
範囲内で精度は約 16 bit、補正なしの 8 bit 推定値は速いが正規化には粗いため、
いずれも採用していない 値は CPU によって異なるので、対象ハードウェアで
`cargo bench` を実行して確認する

## 最小サポート Rust バージョン

Rust 1.87 (`Cargo.toml` の `rust-version`)、CI でライブラリを既定 feature と
全 feature で検査する 1.86 は `AlignedVec` の `Vec::len` / `Vec::as_slice` を
包む `const fn` を受け付けない 開発と CI は `rust-toolchain.toml` で固定した
toolchain を使う

## ビルドとテスト

```sh
cargo test --all-features                                     # unit + doc tests
cargo test --lib --no-default-features --features libm        # no_std code path on the host
cargo build --lib --no-default-features --features libm --target thumbv7em-none-eabihf
cargo clippy --all-targets --all-features -- -D warnings      # pedantic + nursery (Cargo.toml)
cargo bench
scripts/preflight.sh            # every CI step with the same arguments
scripts/preflight.sh --quick    # static checks, builds, docs and `cargo test --lib`
```

## ライセンス

以下のいずれかを選択できる

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE](LICENSE))
