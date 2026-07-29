//! Integer hashing: the addressability substrate.
//!
//! Every value this crate generates is a pure function of a coordinate
//! and a seed, with no stream state in between. That is what makes a
//! texture *addressable*: tile (10⁶, −4) costs the same to evaluate as
//! tile (0, 0), and evaluating it does not depend on which tiles were
//! evaluated before it.
//!
//! It is also what makes the CPU and GPU paths agree. A stream RNG
//! would require reproducing its state machine on the device; a hash
//! requires reproducing four integer operations, and integer operations
//! are exactly specified everywhere.
//!
//! The mixing function is the PCG output permutation
//! `xorshift-multiply-xorshift`, chosen because it avalanches well on
//! low-entropy inputs. Texture coordinates are the worst case for a
//! weak hash: adjacent tiles differ in one low bit, and a hash that
//! does not avalanche produces visible grid artefacts.

/// Mix a single 32-bit word.
///
/// Every constant here is odd, so the multiply is invertible and no
/// input collapses to a fixed point.
#[inline]
pub const fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Hash a 2D integer coordinate against a seed.
///
/// The coordinates are folded in with distinct odd multipliers before
/// mixing, so `(x, y)` and `(y, x)` do not collide and a diagonal walk
/// does not repeat.
#[inline]
pub const fn hash2(x: i32, y: i32, seed: u32) -> u32 {
    let h = (x as u32).wrapping_mul(0x27d4_eb2d)
        ^ (y as u32).wrapping_mul(0x1656_67b1)
        ^ seed.wrapping_mul(0x9e37_79b1);
    mix(h)
}

/// Hash a 3D integer coordinate, for the octave index of a fractal sum.
#[inline]
pub const fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let h = (x as u32).wrapping_mul(0x27d4_eb2d)
        ^ (y as u32).wrapping_mul(0x1656_67b1)
        ^ (z as u32).wrapping_mul(0x0d2f_1b3d)
        ^ seed.wrapping_mul(0x9e37_79b1);
    mix(h)
}

/// A uniform `f32` in `[0, 1)` from a hash.
///
/// Built by placing 23 mantissa bits under an exponent of zero and
/// subtracting one, rather than by dividing. The construction is exact
/// on any IEEE-754 implementation: no rounding happens, so the CPU and
/// the GPU cannot disagree about it.
#[inline]
pub fn unit_f32(h: u32) -> f32 {
    f32::from_bits((h >> 9) | 0x3f80_0000) - 1.0
}

/// A uniform `f32` in `[-1, 1)`.
#[inline]
pub fn signed_f32(h: u32) -> f32 {
    unit_f32(h) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_stays_in_range() {
        for i in 0..100_000u32 {
            let v = unit_f32(mix(i));
            assert!((0.0..1.0).contains(&v), "{v} out of range at {i}");
        }
    }

    #[test]
    fn adjacent_coordinates_decorrelate() {
        // A hash that fails to avalanche shows up as visible grid
        // structure. Adjacent cells should share no more bits than
        // chance would predict: around 16 of 32.
        let mut total = 0u32;
        let n = 10_000i32;
        for i in 0..n {
            let a = hash2(i, 0, 1);
            let b = hash2(i + 1, 0, 1);
            total += (a ^ b).count_ones();
        }
        let mean = total as f64 / n as f64;
        assert!(
            (13.0..19.0).contains(&mean),
            "mean Hamming distance {mean} suggests poor avalanche"
        );
    }

    #[test]
    fn coordinates_are_not_symmetric() {
        // (x, y) and (y, x) colliding would mirror the texture about
        // the diagonal.
        assert_ne!(hash2(3, 7, 0), hash2(7, 3, 0));
    }

    #[test]
    fn the_seed_actually_moves_the_field() {
        assert_ne!(hash2(0, 0, 0), hash2(0, 0, 1));
    }
}
