//! Fixed-point arithmetic, so that exactness stops depending on the
//! driver.
//!
//! The floating-point synthesis path is bit-identical to the device on
//! llvmpipe and is not on Metal, because a shader compiler may fuse a
//! multiply and an add into a single `fma` and WGSL cannot forbid it.
//! Careful `f32` gets the disagreement down to 9e-6 and cannot get it to
//! zero, because the decision is not the program's to make.
//!
//! Integers have no such freedom. `a * b` on two `i32` has one answer,
//! and every driver gives it. This module is therefore the same move
//! `perturbation-kernel` makes underneath: replace the floating-point
//! arithmetic with integer arithmetic and the reproducibility question
//! stops being interesting.
//!
//! # The format
//!
//! Q4.27 in an `i32`: the value is the integer divided by 2²⁷, so the
//! representable range is `[-16, 16)` and the step is 7.45e-9.
//!
//! Four integer bits are not arbitrary. The widest intermediate in the
//! whole synthesis path is `6t - 15` inside the quintic interpolant,
//! which reaches -15, and the contrast remap reaches `contrast` times
//! one. Three bits would clip the first. The 27 fractional bits that
//! remain are finer than an `f32` mantissa anywhere above 0.5, and
//! coarser below it, which is the trade a fixed-point format makes.
//!
//! # Multiplication
//!
//! `mul` needs the full 64-bit product before shifting it back down.
//! Rust has `i64`; WGSL has no integer wider than 32 bits, so the mirror
//! in `synth_exact.wgsl` builds the product from four 16-bit partial
//! products by hand. Both truncate towards zero rather than shifting
//! arithmetically, because truncation is the behaviour that is easy to
//! state and easy to reproduce on a machine with no `i64`.
//!
//! # Division
//!
//! There is none. The only division in the synthesis path is the fractal
//! normalisation `sum / norm`, and `norm` depends only on the octave
//! count. The host computes the reciprocal once, in integer arithmetic,
//! and passes it to the device as another Q4.27 value, so both sides
//! perform a multiply and neither performs a divide.

/// A Q4.27 fixed-point number.
pub type Fx = i32;

/// Fractional bits.
pub const SHIFT: u32 = 27;

/// The value 1.
pub const ONE: Fx = 1 << SHIFT;

/// Convert from `f32`, truncating towards zero.
///
/// Used at the boundary, for parameters the caller supplies as floats.
/// Anything outside `[-16, 16)` saturates rather than wrapping, since a
/// wrapped contrast would be a very confusing picture.
#[inline]
pub fn from_f32(v: f32) -> Fx {
    let scaled = v as f64 * ONE as f64;
    if scaled >= i32::MAX as f64 {
        i32::MAX
    } else if scaled <= i32::MIN as f64 {
        i32::MIN
    } else {
        scaled as i32
    }
}

/// Convert to `f32`.
///
/// Exact when the result has 24 or fewer significant bits, which is
/// every value the synthesis path produces in `[0, 1]` above 2⁻³.
#[inline]
pub fn to_f32(v: Fx) -> f32 {
    v as f32 / ONE as f32
}

/// Multiply, truncating towards zero.
///
/// The WGSL mirror computes the same 64-bit product from 16-bit halves.
#[inline]
pub fn mul(a: Fx, b: Fx) -> Fx {
    let neg = (a < 0) != (b < 0);
    let p = (a.unsigned_abs() as u64) * (b.unsigned_abs() as u64);
    let r = (p >> SHIFT) as i32;
    if neg {
        r.wrapping_neg()
    } else {
        r
    }
}

/// Square root of a non-negative value.
///
/// `sqrt(a / 2²⁷) * 2²⁷` is `isqrt(a * 2²⁷)`, so this is an integer
/// square root of a value up to 2⁵⁷ wide. The bit-by-bit algorithm is
/// used rather than `u64::isqrt` so that this file and the WGSL mirror
/// run the same steps; the mirror has to spell out the 64-bit
/// arithmetic and there is no reason for the two to differ.
#[inline]
pub fn sqrt(a: Fx) -> Fx {
    debug_assert!(a >= 0, "sqrt of a negative fixed-point value");
    isqrt64((a as u64) << SHIFT) as i32
}

/// Floor of the square root of a 64-bit integer.
///
/// Restoring binary square root: one result bit per iteration, highest
/// first. Every operation is an integer compare, subtract or shift, so
/// the WGSL version needs only a 64-bit add, subtract and compare built
/// from `u32` pairs.
fn isqrt64(n: u64) -> u64 {
    let mut num = n;
    let mut res: u64 = 0;
    // Highest even power of two not above `n`.
    let mut bit: u64 = 1u64 << 62;
    while bit > num {
        bit >>= 2;
    }
    while bit != 0 {
        if num >= res + bit {
            num -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    res
}

/// Reciprocal of a positive value, for the host to precompute.
///
/// Deliberately not `#[inline]`-cheap and deliberately not available on
/// the device: the whole point is that this runs once on the host and
/// the device multiplies by the result.
pub fn recip(v: Fx) -> Fx {
    assert!(v > 0, "reciprocal of a non-positive fixed-point value");
    (((ONE as i64) << SHIFT) / v as i64) as i32
}

/// The normalisation divisor a fractal sum of `octaves` octaves needs.
///
/// Amplitudes halve, so the sum of `n` of them is `2 - 2^(1-n)`. Built
/// by repeated halving rather than by a formula, so that it matches the
/// accumulation in the synthesis loop exactly.
pub fn fractal_norm(octaves: u32) -> Fx {
    let mut amp = ONE;
    let mut norm = 0;
    for _ in 0..octaves.max(1) {
        norm += amp;
        amp >>= 1;
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_is_one() {
        assert_eq!(to_f32(ONE), 1.0);
        assert_eq!(from_f32(1.0), ONE);
        assert_eq!(mul(ONE, ONE), ONE);
        assert_eq!(mul(ONE, -ONE), -ONE);
    }

    #[test]
    fn multiplication_is_exactly_the_truncated_product() {
        // The definition, not a tolerance: mul(a, b) is a*b / 2^27
        // truncated towards zero. Checking it against i64 arithmetic
        // pins the contract the WGSL mirror has to reproduce, and does
        // not launder the answer through an f32 on the way.
        //
        // Comparing via to_f32 was the earlier mistake here. A Q4.27
        // value above 1 needs 28 significant bits and an f32 carries 24,
        // so the conversion lost more than the arithmetic did.
        let cases = [
            0,
            1,
            -1,
            7,
            ONE,
            -ONE,
            ONE - 1,
            ONE / 3,
            3 * ONE,
            -15 * ONE,
            123_456_789,
        ];
        for &a in &cases {
            for &b in &cases {
                let want = ((a as i64 * b as i64) / ONE as i64) as i32;
                assert_eq!(mul(a, b), want, "mul({a}, {b})");
            }
        }
    }

    #[test]
    fn multiplication_tracks_the_real_product() {
        // Same check in real terms, in raw units so that nothing is
        // rounded on the way to the comparison. Three truncations
        // contribute, and the two operand conversions are scaled by the
        // *other* operand, so the bound carries the magnitudes.
        for &a in &[0.0f64, 0.25, -0.5, 1.0, -1.0, 3.75, -15.5, 0.1234] {
            for &b in &[0.0f64, 0.5, -0.75, 1.0, 2.0, -3.0, 0.987] {
                if (a * b).abs() >= 16.0 {
                    continue;
                }
                // Reference from the f32 values that are actually
                // multiplied, not from the f64 literals. 0.987f32 is not
                // 0.987, and at 15.5 that gap alone is 24 raw steps,
                // which is larger than anything the fixed-point path
                // contributes and would be blamed on it.
                let (af, bf) = (a as f32 as f64, b as f32 as f64);
                let got = mul(from_f32(a as f32), from_f32(b as f32)) as f64;
                let want = af * bf * ONE as f64;
                let bound = af.abs() + bf.abs() + 2.0;
                assert!(
                    (got - want).abs() <= bound,
                    "{af} * {bf}: {got} raw, want {want:.1}, off by more than {bound:.1} steps"
                );
            }
        }
    }

    #[test]
    fn multiplication_is_sign_symmetric() {
        // Truncation towards zero, so negating an operand negates the
        // result exactly. The WGSL mirror relies on this: it multiplies
        // magnitudes and applies the sign afterwards.
        for a in [1, 7, 12345, ONE, ONE - 1, 3 * ONE] {
            for b in [1, 3, 99999, ONE, ONE / 3] {
                assert_eq!(mul(a, b), -mul(-a, b));
                assert_eq!(mul(a, b), -mul(a, -b));
                assert_eq!(mul(a, b), mul(-a, -b));
            }
        }
    }

    #[test]
    fn square_root_is_exact_on_squares() {
        // Only up to 3: the format tops out at 16, so 4 squared is
        // already the last representable square and 5 squared is not
        // representable at all.
        for v in [0, 1, 2, 3] {
            let x = v * ONE;
            assert_eq!(sqrt(mul(x, x)), x, "sqrt of {v} squared");
        }
        // Halves too, where the fractional bits matter.
        for x in [ONE / 2, ONE / 4, 3 * ONE / 2] {
            assert_eq!(sqrt(mul(x, x)), x, "sqrt of ({x} / 2^27) squared");
        }
    }

    #[test]
    fn square_root_is_the_exact_floor_of_the_real_root() {
        // Verified by squaring rather than by comparing against
        // f64::sqrt: the argument to the integer root reaches 2^57 and
        // an f64 holds only 53 bits of integer exactly, so the reference
        // would be the less accurate of the two.
        for &v in &[
            0,
            1,
            2,
            3,
            12345,
            ONE / 4,
            ONE,
            2 * ONE,
            8 * ONE,
            15 * ONE + 7,
        ] {
            let r = sqrt(v) as u128;
            let n = (v as u128) << SHIFT;
            assert!(r * r <= n, "sqrt({v}) = {r} is too large");
            assert!((r + 1) * (r + 1) > n, "sqrt({v}) = {r} is too small");
        }
    }

    #[test]
    fn isqrt_matches_the_reference_at_the_boundaries() {
        // Perfect squares and the values either side of them, where an
        // off-by-one in the restoring loop would show.
        for k in [0u64, 1, 2, 3, 65535, 65536, 1 << 20, (1u64 << 28) - 1] {
            let sq = k * k;
            assert_eq!(isqrt64(sq), k, "isqrt of {k}^2");
            if sq > 0 {
                assert_eq!(isqrt64(sq - 1), k - 1, "isqrt of {k}^2 - 1");
            }
            assert_eq!(isqrt64(sq + 2 * k), k, "isqrt just below {}^2", k + 1);
        }
    }

    #[test]
    fn the_fractal_norm_matches_the_accumulation_it_stands_for() {
        assert_eq!(fractal_norm(1), ONE);
        assert_eq!(fractal_norm(2), ONE + ONE / 2);
        assert_eq!(fractal_norm(3), ONE + ONE / 2 + ONE / 4);
        // 2 - 2^(1-n), approached but never reached.
        assert!(fractal_norm(20) < 2 * ONE);
    }

    #[test]
    fn reciprocal_undoes_multiplication() {
        for n in 1..=8u32 {
            let norm = fractal_norm(n);
            let r = recip(norm);
            // Round trip within a couple of steps: two truncations.
            let back = mul(norm, r);
            assert!(
                (back - ONE).abs() <= 2,
                "octaves {n}: norm * recip = {back}, want {ONE}"
            );
        }
    }
}
