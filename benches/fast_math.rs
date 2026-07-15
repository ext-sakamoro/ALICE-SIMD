// fast_math bench: aarch64 explicit NEON vs current fallback

use alice_simd::fast_math::{fast_inv_sqrt, fast_rcp};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// aarch64 explicit NEON reciprocal (frecpe + 1 frecps refinement, ~16-bit precision)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_rcp(x: f32) -> f32 {
    use core::arch::aarch64::{vdup_n_f32, vget_lane_f32, vrecpe_f32, vrecps_f32};
    // SAFETY: aarch64 intrinsics unconditional on this arch
    unsafe {
        let v = vdup_n_f32(x);
        let e = vrecpe_f32(v);
        let step = vrecps_f32(v, e);
        vget_lane_f32::<0>(vmul_f32(e, step))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_rcp_no_refine(x: f32) -> f32 {
    use core::arch::aarch64::{vdup_n_f32, vget_lane_f32, vrecpe_f32};
    // SAFETY: aarch64 intrinsics unconditional on this arch
    unsafe {
        let v = vdup_n_f32(x);
        let e = vrecpe_f32(v);
        vget_lane_f32::<0>(e)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_inv_sqrt(x: f32) -> f32 {
    use core::arch::aarch64::{vdup_n_f32, vget_lane_f32, vmul_f32, vrsqrte_f32, vrsqrts_f32};
    // SAFETY: aarch64 intrinsics unconditional on this arch
    unsafe {
        let v = vdup_n_f32(x);
        let e = vrsqrte_f32(v);
        let step = vrsqrts_f32(vmul_f32(e, e), v);
        vget_lane_f32::<0>(vmul_f32(e, step))
    }
}

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vmul_f32;

fn bench_rcp(c: &mut Criterion) {
    let vals: Vec<f32> = (1..=1024).map(|i| i as f32 * 0.1).collect();

    c.bench_function("fast_rcp (non-x86: 1.0/x)", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for &x in &vals {
                sum += fast_rcp(black_box(x));
            }
            sum
        })
    });

    c.bench_function("baseline: 1.0/x direct", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for &x in &vals {
                sum += 1.0_f32 / black_box(x);
            }
            sum
        })
    });

    #[cfg(target_arch = "aarch64")]
    c.bench_function("neon_rcp (frecpe + 1 frecps, ~16-bit)", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for &x in &vals {
                sum += neon_rcp(black_box(x));
            }
            sum
        })
    });

    #[cfg(target_arch = "aarch64")]
    c.bench_function("neon_rcp_no_refine (frecpe only, ~8-bit)", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for &x in &vals {
                sum += neon_rcp_no_refine(black_box(x));
            }
            sum
        })
    });
}

fn bench_inv_sqrt(c: &mut Criterion) {
    let vals: Vec<f32> = (1..=1024).map(|i| i as f32 * 0.1).collect();

    c.bench_function("fast_inv_sqrt (non-x86: 1.0/x.sqrt())", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for &x in &vals {
                sum += fast_inv_sqrt(black_box(x));
            }
            sum
        })
    });

    c.bench_function("baseline: 1.0/x.sqrt() direct", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for &x in &vals {
                sum += 1.0_f32 / black_box(x).sqrt();
            }
            sum
        })
    });

    #[cfg(target_arch = "aarch64")]
    c.bench_function("neon_inv_sqrt (frsqrte + 1 frsqrts, ~16-bit)", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for &x in &vals {
                sum += neon_inv_sqrt(black_box(x));
            }
            sum
        })
    });
}

criterion_group!(benches, bench_rcp, bench_inv_sqrt);
criterion_main!(benches);
