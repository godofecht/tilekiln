//! Noise primitives.
//!
//! # Why every operation here is add, subtract or multiply
//!
//! The synthesis path is mirrored in WGSL so a texture can be built on
//! a compute device. WGSL specifies the accuracy of `+`, `-` and `*` on
//! `f32` exactly, and leaves `sin`, `exp`, `log`, `pow` and friends to
//! the driver. One transcendental anywhere in the pipeline and the CPU
//! and GPU results diverge in the low bits, which is the difference
//! between "the same texture" and "a texture that looks the same".
//!
//! So: gradients come from a hashed lookup table rather than from
//! `sin`/`cos`, the interpolant is a quintic polynomial, and the
//! fractal sum uses power-of-two lacunarity so the scale factors are
//! exact. `tests/gpu.rs` asserts the two paths agree bit for bit.
//!
//! The one deliberate exception is [`Worley`] distance, which needs a
//! square root. `f32::sqrt` *is* correctly rounded in IEEE-754 and WGSL
//! inherits that, so it is safe; `inverseSqrt` would not be.

use crate::hash::{hash2, hash3, signed_f32};

/// Quintic interpolant, `6t⁵ − 15t⁴ + 10t³`.
///
/// Preferred over the cubic `3t² − 2t³` because its second derivative
/// vanishes at both ends, which removes the creases visible along cell
/// boundaries when a gradient field is used as a normal map.
#[inline]
pub fn smootherstep(t: f32) -> f32 {
    // Horner form: fewer roundings than the expanded polynomial, and
    // identical to the WGSL mirror.
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    // `a + t*(b - a)` rather than `(1-t)*a + t*b`: one fewer rounding,
    // and exact at both endpoints.
    a + t * (b - a)
}

/// One of eight unit-ish gradients, selected by hash.
///
/// A table avoids `sin`/`cos` entirely. Eight directions is enough for
/// gradient noise at texture scale and keeps the selection a 3-bit
/// mask. The diagonals are left unnormalised at length √2, exactly as
/// Perlin's improved noise does, because normalising would introduce a
/// division and buy nothing visible.
#[inline]
fn gradient(h: u32) -> (f32, f32) {
    match h & 7 {
        0 => (1.0, 0.0),
        1 => (-1.0, 0.0),
        2 => (0.0, 1.0),
        3 => (0.0, -1.0),
        4 => (1.0, 1.0),
        5 => (-1.0, 1.0),
        6 => (1.0, -1.0),
        _ => (-1.0, -1.0),
    }
}

/// Value noise: hashed scalars at lattice points, quintically blended.
///
/// Cheaper than gradient noise and blockier. Useful as a warp source
/// where the character matters less than the cost.
pub fn value2(x: f64, y: f64, seed: u32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = (x - xi) as f32;
    let yf = (y - yi) as f32;
    let (ix, iy) = (xi as i64 as i32, yi as i64 as i32);

    let u = smootherstep(xf);
    let v = smootherstep(yf);

    let c00 = signed_f32(hash2(ix, iy, seed));
    let c10 = signed_f32(hash2(ix + 1, iy, seed));
    let c01 = signed_f32(hash2(ix, iy + 1, seed));
    let c11 = signed_f32(hash2(ix + 1, iy + 1, seed));

    lerp(lerp(c00, c10, u), lerp(c01, c11, u), v)
}

/// Gradient (Perlin) noise.
///
/// Returns roughly `[-1, 1]`. The classic construction: a hashed
/// gradient per lattice corner, dotted with the offset to the sample
/// point, blended quintically.
pub fn gradient2(x: f64, y: f64, seed: u32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = (x - xi) as f32;
    let yf = (y - yi) as f32;
    let (ix, iy) = (xi as i64 as i32, yi as i64 as i32);

    let u = smootherstep(xf);
    let v = smootherstep(yf);

    let dot = |cx: i32, cy: i32, dx: f32, dy: f32| {
        let (gx, gy) = gradient(hash2(ix + cx, iy + cy, seed));
        gx * dx + gy * dy
    };

    let n00 = dot(0, 0, xf, yf);
    let n10 = dot(1, 0, xf - 1.0, yf);
    let n01 = dot(0, 1, xf, yf - 1.0);
    let n11 = dot(1, 1, xf - 1.0, yf - 1.0);

    lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
}

/// Worley (cellular) noise: distance to the nearest feature point.
///
/// One jittered point per cell, searched over the 3×3 neighbourhood.
/// Returns the Euclidean distance, which is unbounded above in
/// principle and below about 1.5 in practice.
pub fn worley2(x: f64, y: f64, seed: u32) -> f32 {
    let xi = x.floor() as i64 as i32;
    let yi = y.floor() as i64 as i32;
    let mut best = f32::MAX;

    for oy in -1..=1i32 {
        for ox in -1..=1i32 {
            let cx = xi + ox;
            let cy = yi + oy;
            let h = hash2(cx, cy, seed);
            // Two independent offsets from one hash: the low and high
            // halves are decorrelated by the mixing function.
            let px = cx as f64 + (signed_f32(h) * 0.5 + 0.5) as f64;
            let py = cy as f64 + (signed_f32(hash3(cx, cy, 1, seed)) * 0.5 + 0.5) as f64;
            let dx = (px - x) as f32;
            let dy = (py - y) as f32;
            let d2 = dx * dx + dy * dy;
            if d2 < best {
                best = d2;
            }
        }
    }
    // sqrt is correctly rounded in IEEE-754 and WGSL inherits that, so
    // this is the one root the exactness argument tolerates.
    best.sqrt()
}

/// Which primitive a fractal sum is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Hashed scalars at lattice points. Cheap, blocky.
    Value,
    /// Perlin gradient noise. Smoother, the usual default.
    Gradient,
    /// Distance to the nearest feature point. Cellular.
    Worley,
}

impl Basis {
    #[inline]
    /// Evaluate this primitive at `(x, y)`.
    pub fn sample(self, x: f64, y: f64, seed: u32) -> f32 {
        match self {
            Basis::Value => value2(x, y, seed),
            Basis::Gradient => gradient2(x, y, seed),
            // Recentre to roughly [-1, 1] so octaves compose sensibly.
            Basis::Worley => worley2(x, y, seed) * 2.0 - 1.0,
        }
    }
}

/// Fractal Brownian motion: octaves at doubling frequency and halving
/// amplitude.
///
/// Lacunarity is fixed at 2 and gain at 0.5 on purpose. Both are exact
/// in binary floating point, so the octave scale factors introduce no
/// rounding of their own and the CPU and GPU sums stay in lockstep. A
/// configurable lacunarity of, say, 2.1 would reintroduce exactly the
/// divergence this crate is built to avoid.
pub fn fbm(basis: Basis, x: f64, y: f64, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut norm = 0.0f32;
    let mut fx = x;
    let mut fy = y;

    for o in 0..octaves.max(1) {
        sum += amp * basis.sample(fx, fy, seed.wrapping_add(o.wrapping_mul(0x9e37_79b1)));
        norm += amp;
        amp *= 0.5;
        fx *= 2.0;
        fy *= 2.0;
    }
    // Dividing by the accumulated amplitude keeps the result in the
    // basis's own range regardless of octave count. `norm` is a sum of
    // negative powers of two, so the division is exact.
    sum / norm
}

/// Ridged multifractal: `1 − |n|`, sharpened, applied **per octave**.
///
/// The ridging has to happen inside the octave loop. Applying it to the
/// finished fractal sum instead produces mush: the sum is already
/// smooth and centred near zero, so `1 − |sum|` is a broad blob rather
/// than a crease. Ridging each octave before accumulating is what puts
/// a sharp crest at every zero crossing at every scale, which is what
/// makes the result read as rock rather than as cloud.
///
/// Returns a value in `[0, 1]`, unlike [`fbm`], which is signed. The
/// caller must not remap it a second time.
pub fn ridged(basis: Basis, x: f64, y: f64, octaves: u32, sharpness: u32, seed: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut norm = 0.0f32;
    let mut fx = x;
    let mut fy = y;

    for o in 0..octaves.max(1) {
        let n = basis.sample(fx, fy, seed.wrapping_add(o.wrapping_mul(0x9e37_79b1)));
        let r = 1.0 - n.abs();
        // Integer exponent by repeated multiplication: `powf` is not
        // exactly specified in WGSL, and the mirror has to match.
        let mut sharp = r;
        for _ in 1..sharpness.max(1) {
            sharp *= r;
        }
        sum += amp * sharp;
        norm += amp;
        amp *= 0.5;
        fx *= 2.0;
        fy *= 2.0;
    }
    sum / norm
}

/// Domain warp: displace the sample point by another noise field.
///
/// The classic trick for turning bland noise into something that looks
/// eroded or marbled. `strength` is in the same units as the input
/// coordinates.
pub fn warped(basis: Basis, x: f64, y: f64, octaves: u32, strength: f32, seed: u32) -> f32 {
    let wx = fbm(Basis::Value, x + 5.2, y + 1.3, 2, seed ^ 0x5f35_6495);
    let wy = fbm(Basis::Value, x + 1.7, y + 9.2, 2, seed ^ 0x3c6e_f372);
    fbm(
        basis,
        x + (strength * wx) as f64,
        y + (strength * wy) as f64,
        octaves,
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smootherstep_pins_its_endpoints() {
        assert_eq!(smootherstep(0.0), 0.0);
        assert_eq!(smootherstep(1.0), 1.0);
        assert!((smootherstep(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gradient_noise_vanishes_on_the_lattice() {
        // Perlin noise is zero at integer coordinates by construction:
        // the offset vector is zero, so every dot product is zero.
        for i in -5..5 {
            for j in -5..5 {
                let v = gradient2(i as f64, j as f64, 7);
                assert!(v.abs() < 1e-6, "expected 0 at ({i}, {j}), got {v}");
            }
        }
    }

    #[test]
    fn noise_stays_in_a_sane_range() {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for i in 0..20_000 {
            let x = i as f64 * 0.0137;
            let y = i as f64 * 0.0271;
            let v = fbm(Basis::Gradient, x, y, 4, 1);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(
            lo > -1.5 && hi < 1.5,
            "fbm range [{lo}, {hi}] is wider than expected"
        );
    }

    #[test]
    fn evaluation_is_addressable_not_sequential() {
        // The whole point: a far-away sample costs nothing extra and
        // does not depend on what was sampled before it.
        let a = fbm(Basis::Gradient, 1_000_000.5, -4_000_000.25, 4, 3);
        let b = fbm(Basis::Gradient, 1_000_000.5, -4_000_000.25, 4, 3);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn worley_is_non_negative() {
        for i in 0..5_000 {
            let x = i as f64 * 0.017;
            let v = worley2(x, x * 0.3, 11);
            assert!(v >= 0.0, "distance {v} is negative");
        }
    }
}
