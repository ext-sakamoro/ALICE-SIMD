//! Packed Vec3/Vec4 operations for ALICE ecosystem.
//!
//! Lightweight vector types with SIMD-friendly layout.
//! Uses `fast_inv_sqrt` for high-performance normalization.

use crate::fast_math::{fast_inv_sqrt, fma, lerp};

// ── Vec3 ───────────────────────────────────────────────────────────────

/// 3D vector (f32).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// ゼロベクトル。
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// 単位ベクトル (1,1,1)。
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    /// X 軸単位ベクトル。
    pub const UNIT_X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// Y 軸単位ベクトル。
    pub const UNIT_Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Z 軸単位ベクトル。
    pub const UNIT_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// 新規作成。
    #[inline(always)]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// 全成分同一値。
    #[inline(always)]
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// 加算。
    #[inline(always)]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }

    /// 減算。
    #[inline(always)]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }

    /// 成分ごとの乗算。
    #[inline(always)]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }

    /// スカラー倍。
    #[inline(always)]
    #[must_use]
    pub const fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// 成分ごとの否定。
    #[inline(always)]
    #[must_use]
    pub const fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// 内積。
    #[inline(always)]
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        fma(self.x, rhs.x, fma(self.y, rhs.y, self.z * rhs.z))
    }

    /// 外積。
    #[inline(always)]
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: fma(self.y, rhs.z, -(self.z * rhs.y)),
            y: fma(self.z, rhs.x, -(self.x * rhs.z)),
            z: fma(self.x, rhs.y, -(self.y * rhs.x)),
        }
    }

    /// 長さの二乗。
    #[inline(always)]
    #[must_use]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// 長さ。
    #[inline(always)]
    #[must_use]
    pub fn length(self) -> f32 {
        let sq = self.length_sq();
        sq * fast_inv_sqrt(sq)
    }

    /// 高速正規化（`fast_inv_sqrt` 使用）。
    ///
    /// ゼロベクトルの場合はゼロベクトルを返す。
    #[inline(always)]
    #[must_use]
    pub fn normalize(self) -> Self {
        let sq = self.length_sq();
        if sq < 1e-12 {
            return Self::ZERO;
        }
        self.scale(fast_inv_sqrt(sq))
    }

    /// 線形補間。
    #[inline(always)]
    #[must_use]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self {
            x: lerp(self.x, rhs.x, t),
            y: lerp(self.y, rhs.y, t),
            z: lerp(self.z, rhs.z, t),
        }
    }

    /// 成分ごとの最小値。
    #[inline(always)]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self {
            x: if self.x < rhs.x { self.x } else { rhs.x },
            y: if self.y < rhs.y { self.y } else { rhs.y },
            z: if self.z < rhs.z { self.z } else { rhs.z },
        }
    }

    /// 成分ごとの最大値。
    #[inline(always)]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self {
            x: if self.x > rhs.x { self.x } else { rhs.x },
            y: if self.y > rhs.y { self.y } else { rhs.y },
            z: if self.z > rhs.z { self.z } else { rhs.z },
        }
    }

    /// 二点間の距離の二乗。
    #[inline(always)]
    #[must_use]
    pub fn distance_sq(self, rhs: Self) -> f32 {
        self.sub(rhs).length_sq()
    }

    /// 配列への変換。
    #[inline(always)]
    #[must_use]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// 配列からの変換。
    #[inline(always)]
    #[must_use]
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }
}

// ── Vec4 ───────────────────────────────────────────────────────────────

/// 4D vector (f32). SIMD 128-bit 幅に自然にアラインする。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    /// ゼロベクトル。
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// 新規作成。
    #[inline(always)]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// 全成分同一値。
    #[inline(always)]
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self {
            x: v,
            y: v,
            z: v,
            w: v,
        }
    }

    /// `Vec3` + w 成分から生成。
    #[inline(always)]
    #[must_use]
    pub const fn from_vec3(v: Vec3, w: f32) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w,
        }
    }

    /// xyz 成分を `Vec3` として取得。
    #[inline(always)]
    #[must_use]
    pub const fn xyz(self) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    /// 加算。
    #[inline(always)]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }

    /// 減算。
    #[inline(always)]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }

    /// スカラー倍。
    #[inline(always)]
    #[must_use]
    pub const fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
            w: self.w * s,
        }
    }

    /// 内積。
    #[inline(always)]
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        fma(
            self.x,
            rhs.x,
            fma(self.y, rhs.y, fma(self.z, rhs.z, self.w * rhs.w)),
        )
    }

    /// 長さの二乗。
    #[inline(always)]
    #[must_use]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// 高速正規化。
    #[inline(always)]
    #[must_use]
    pub fn normalize(self) -> Self {
        let sq = self.length_sq();
        if sq < 1e-12 {
            return Self::ZERO;
        }
        self.scale(fast_inv_sqrt(sq))
    }

    /// 配列への変換。
    #[inline(always)]
    #[must_use]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// 配列からの変換。
    #[inline(always)]
    #[must_use]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
            w: a[3],
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.01;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    // ── Vec3 テスト ────────────────────────────────────────────────

    #[test]
    fn vec3_new() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn vec3_splat() {
        let v = Vec3::splat(5.0);
        assert_eq!(v, Vec3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn vec3_add() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a.add(b);
        assert_eq!(c, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn vec3_sub() {
        let a = Vec3::new(5.0, 7.0, 9.0);
        let b = Vec3::new(1.0, 2.0, 3.0);
        let c = a.sub(b);
        assert_eq!(c, Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn vec3_mul() {
        let a = Vec3::new(2.0, 3.0, 4.0);
        let b = Vec3::new(5.0, 6.0, 7.0);
        let c = a.mul(b);
        assert_eq!(c, Vec3::new(10.0, 18.0, 28.0));
    }

    #[test]
    fn vec3_scale() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let s = v.scale(2.0);
        assert_eq!(s, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn vec3_neg() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let n = v.neg();
        assert_eq!(n, Vec3::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        // 1*4 + 2*5 + 3*6 = 32
        assert!(approx_eq(a.dot(b), 32.0));
    }

    #[test]
    fn vec3_cross() {
        let x = Vec3::UNIT_X;
        let y = Vec3::UNIT_Y;
        let z = x.cross(y);
        assert!(vec3_approx_eq(z, Vec3::UNIT_Z));
    }

    #[test]
    fn vec3_cross_anticommutative() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let ab = a.cross(b);
        let ba = b.cross(a);
        assert!(vec3_approx_eq(ab, ba.neg()));
    }

    #[test]
    fn vec3_length_sq() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx_eq(v.length_sq(), 25.0));
    }

    #[test]
    fn vec3_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx_eq(v.length(), 5.0));
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!(approx_eq(n.length(), 1.0));
        assert!(approx_eq(n.x, 0.6));
        assert!(approx_eq(n.y, 0.8));
    }

    #[test]
    fn vec3_normalize_zero() {
        let n = Vec3::ZERO.normalize();
        assert_eq!(n, Vec3::ZERO);
    }

    #[test]
    fn vec3_lerp() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 20.0, 30.0);
        let mid = a.lerp(b, 0.5);
        assert!(vec3_approx_eq(mid, Vec3::new(5.0, 10.0, 15.0)));
    }

    #[test]
    fn vec3_lerp_endpoints() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(vec3_approx_eq(a.lerp(b, 0.0), a));
        assert!(vec3_approx_eq(a.lerp(b, 1.0), b));
    }

    #[test]
    fn vec3_min_max() {
        let a = Vec3::new(1.0, 5.0, 3.0);
        let b = Vec3::new(4.0, 2.0, 6.0);
        assert_eq!(a.min(b), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a.max(b), Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn vec3_distance_sq() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(4.0, 0.0, 0.0);
        assert!(approx_eq(a.distance_sq(b), 9.0));
    }

    #[test]
    fn vec3_to_from_array() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let arr = v.to_array();
        assert_eq!(arr, [1.0, 2.0, 3.0]);
        assert_eq!(Vec3::from_array(arr), v);
    }

    #[test]
    fn vec3_constants() {
        assert_eq!(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(Vec3::ONE, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(Vec3::UNIT_X, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(Vec3::UNIT_Y, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(Vec3::UNIT_Z, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn vec3_dot_perpendicular() {
        assert!(approx_eq(Vec3::UNIT_X.dot(Vec3::UNIT_Y), 0.0));
        assert!(approx_eq(Vec3::UNIT_Y.dot(Vec3::UNIT_Z), 0.0));
    }

    #[test]
    fn vec3_dot_parallel() {
        assert!(approx_eq(Vec3::UNIT_X.dot(Vec3::UNIT_X), 1.0));
    }

    // ── Vec4 テスト ────────────────────────────────────────────────

    #[test]
    fn vec4_new() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn vec4_splat() {
        let v = Vec4::splat(7.0);
        assert_eq!(v, Vec4::new(7.0, 7.0, 7.0, 7.0));
    }

    #[test]
    fn vec4_from_vec3() {
        let v3 = Vec3::new(1.0, 2.0, 3.0);
        let v4 = Vec4::from_vec3(v3, 1.0);
        assert_eq!(v4, Vec4::new(1.0, 2.0, 3.0, 1.0));
    }

    #[test]
    fn vec4_xyz() {
        let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v4.xyz(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn vec4_add() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a.add(b), Vec4::new(6.0, 8.0, 10.0, 12.0));
    }

    #[test]
    fn vec4_sub() {
        let a = Vec4::new(5.0, 6.0, 7.0, 8.0);
        let b = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a.sub(b), Vec4::new(4.0, 4.0, 4.0, 4.0));
    }

    #[test]
    fn vec4_scale() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.scale(2.0), Vec4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn vec4_dot() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        // 1*5 + 2*6 + 3*7 + 4*8 = 70
        assert!(approx_eq(a.dot(b), 70.0));
    }

    #[test]
    fn vec4_length_sq() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        // 1+4+9+16=30
        assert!(approx_eq(v.length_sq(), 30.0));
    }

    #[test]
    fn vec4_normalize() {
        let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let n = v.normalize();
        assert!(approx_eq(n.x, 1.0));
        assert!(approx_eq(n.length_sq(), 1.0));
    }

    #[test]
    fn vec4_normalize_zero() {
        let n = Vec4::ZERO.normalize();
        assert_eq!(n, Vec4::ZERO);
    }

    #[test]
    fn vec4_to_from_array() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let arr = v.to_array();
        assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Vec4::from_array(arr), v);
    }

    #[test]
    fn vec4_zero_constant() {
        assert_eq!(Vec4::ZERO, Vec4::new(0.0, 0.0, 0.0, 0.0));
    }
}
