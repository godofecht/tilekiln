//! The synthesis path in fixed point, identical on every driver.
//!
//! [`crate::noise`] is the floating-point path. It is bit-identical to
//! the device on llvmpipe and within 9e-6 on Metal, and the difference
//! is not something the program controls: a shader compiler may fuse a
//! multiply and an add into a single `fma`, and WGSL has no way to say
//! no.
//!
//! This module removes the question by removing the floats. Every
//! operation is an `i32` multiply, add or shift in the Q4.27 format from
//! [`crate::fixed`], and integer arithmetic has one answer. There is
//! nothing for a compiler to reassociate, nothing to contract, and no
//! `fma` to substitute.
//!
//! The construction is otherwise the same: quintic interpolant, eight
//! hashed gradients, lacunarity two, coordinates split into an integer
//! cell and a fractional offset. The pictures are the same pictures. The
//! low bits differ from the floating-point path, because 27 fractional
//! bits are not 24 significant ones, so the two are alternatives rather
//! than a fast version and a slow version of one thing.
//!
//! # What this costs
//!
//! A multiply becomes a 64-bit product built from four 16-bit partial
//! products, and Worley distance becomes a 31-iteration integer square
//! root. Measured on one core over a 256×256 tile, five octaves:
//!
//! | material | float | fixed | ratio |
//! |---|---|---|---|
//! | value / fractal | 4.2 ms | 6.7 ms | 1.6x |
//! | worley / fractal | 12.4 ms | 27.5 ms | 2.2x |
//! | gradient / warped | 8.2 ms | 26.5 ms | 3.2x |
//! | gradient / ridged | 2.9 ms | 12.2 ms | 4.2x |
//!
//! Ridged is the worst case because sharpening runs the emulated
//! multiply `sharpness` times per octave, where the floating-point path
//! runs a hardware multiply. On a device the ratio matters less than it
//! reads, since the work is per pixel and embarrassingly parallel.
//!
//! # What it does not fix
//!
//! The lattice cell is still an `i32` and still wraps, so
//! [`crate::Material::max_tile`] applies here exactly as it does to the
//! floating-point path.

use crate::fixed::{self, mul, Fx, ONE, SHIFT};
use crate::hash::{hash2, hash3};
use crate::material::{Material, Pattern, Tile, TileId};
use crate::noise::Basis;

/// A coordinate split into an integer cell and a Q4.27 offset.
///
/// The fixed-point counterpart of [`crate::noise::Lattice`]. `frac` is
/// always in `[0, ONE)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lat {
    /// Integer cell index, wrapping at 2³².
    pub cell: i32,
    /// Offset within the cell, in `[0, ONE)`.
    pub frac: Fx,
}

impl Lat {
    /// Split a real coordinate.
    ///
    /// The only `f64` in the module, and it runs once per tile rather
    /// than once per pixel: the device is handed the result.
    pub fn split(v: f64) -> Self {
        let fl = v.floor();
        Self {
            cell: fl as i64 as i32,
            frac: ((v - fl) * ONE as f64) as i32,
        }
    }

    /// Re-normalise after `frac` has left `[0, ONE)`.
    ///
    /// An arithmetic shift floors, which is what a lattice needs, and
    /// `frac - (carry << SHIFT)` then lands back in range for any input.
    #[inline]
    pub fn renormalise(self) -> Self {
        let carry = self.frac >> SHIFT;
        Self {
            cell: self.cell.wrapping_add(carry),
            frac: self.frac - (carry << SHIFT),
        }
    }

    /// Advance by `delta` cells.
    #[inline]
    pub fn offset(self, delta: Fx) -> Self {
        Self {
            cell: self.cell,
            frac: self.frac + delta,
        }
        .renormalise()
    }

    /// Advance by `n` steps of `step`, where the total may be large.
    ///
    /// Used to walk across a tile. The product needs more than 32 bits
    /// once the tile is wide and the frequency is high, so it is taken
    /// at full width and split, rather than being computed in the
    /// format and overflowing it.
    #[inline]
    pub fn step(self, n: u32, step: Fx) -> Self {
        let p = n as i64 * step as i64;
        let cell_delta = (p >> SHIFT) as i32;
        let frac_delta = (p & (ONE as i64 - 1)) as i32;
        Self {
            cell: self.cell.wrapping_add(cell_delta),
            frac: self.frac + frac_delta,
        }
        .renormalise()
    }

    /// Double the coordinate, as one fractal octave does.
    #[inline]
    pub fn double(self) -> Self {
        let d = self.frac << 1;
        let carry = d >> SHIFT;
        Self {
            cell: self.cell.wrapping_mul(2).wrapping_add(carry),
            frac: d - (carry << SHIFT),
        }
    }
}

/// A hashed value in `[-1, 1)`.
///
/// Shifts only. The 23 bits an `f32` mantissa would have carried land
/// directly in the top of the fixed-point fraction, so this is the same
/// value [`crate::hash::signed_f32`] produces, exactly, with no
/// conversion.
#[inline]
fn signed(h: u32) -> Fx {
    (((h >> 9) << (SHIFT - 22)) as i32).wrapping_sub(ONE)
}

/// Quintic interpolant `6t⁵ − 15t⁴ + 10t³`, in Horner form.
///
/// `6t - 15` reaches -15, which is why the format keeps four integer
/// bits.
#[inline]
pub fn smootherstep(t: Fx) -> Fx {
    let t2 = mul(t, t);
    let t3 = mul(t2, t);
    let inner = mul(t, 6 * ONE) - 15 * ONE;
    let outer = mul(t, inner) + 10 * ONE;
    mul(t3, outer)
}

#[inline]
fn lerp(a: Fx, b: Fx, t: Fx) -> Fx {
    a + mul(t, b - a)
}

/// One of eight gradients, as exact fixed-point components.
///
/// Every component is 0 or ±1, so the dot product below is exact: a
/// multiply by `ONE` shifts back to itself.
#[inline]
fn gradient(h: u32) -> (Fx, Fx) {
    match h & 7 {
        0 => (ONE, 0),
        1 => (-ONE, 0),
        2 => (0, ONE),
        3 => (0, -ONE),
        4 => (ONE, ONE),
        5 => (-ONE, ONE),
        6 => (ONE, -ONE),
        _ => (-ONE, -ONE),
    }
}

/// Value noise.
pub fn value2(lx: Lat, ly: Lat, seed: u32) -> Fx {
    let (ix, iy) = (lx.cell, ly.cell);
    let u = smootherstep(lx.frac);
    let v = smootherstep(ly.frac);

    let c00 = signed(hash2(ix, iy, seed));
    let c10 = signed(hash2(ix.wrapping_add(1), iy, seed));
    let c01 = signed(hash2(ix, iy.wrapping_add(1), seed));
    let c11 = signed(hash2(ix.wrapping_add(1), iy.wrapping_add(1), seed));

    lerp(lerp(c00, c10, u), lerp(c01, c11, u), v)
}

/// Gradient (Perlin) noise.
pub fn gradient2(lx: Lat, ly: Lat, seed: u32) -> Fx {
    let (ix, iy) = (lx.cell, ly.cell);
    let (xf, yf) = (lx.frac, ly.frac);
    let u = smootherstep(xf);
    let v = smootherstep(yf);

    let dot = |cx: i32, cy: i32, dx: Fx, dy: Fx| {
        let (gx, gy) = gradient(hash2(ix.wrapping_add(cx), iy.wrapping_add(cy), seed));
        mul(gx, dx) + mul(gy, dy)
    };

    let n00 = dot(0, 0, xf, yf);
    let n10 = dot(1, 0, xf - ONE, yf);
    let n01 = dot(0, 1, xf, yf - ONE);
    let n11 = dot(1, 1, xf - ONE, yf - ONE);

    lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
}

/// Worley (cellular) noise: distance to the nearest feature point.
pub fn worley2(lx: Lat, ly: Lat, seed: u32) -> Fx {
    let (xi, yi) = (lx.cell, ly.cell);
    // Larger than any distance reachable in a 3x3 search.
    let mut best = 15 * ONE;

    for oy in -1..=1i32 {
        for ox in -1..=1i32 {
            let cx = xi.wrapping_add(ox);
            let cy = yi.wrapping_add(oy);
            // Jitter into [0, 1): halving a value that is a multiple of
            // 32 raw units, so the shift discards nothing.
            let jx = (signed(hash2(cx, cy, seed)) >> 1) + ONE / 2;
            let jy = (signed(hash3(cx, cy, 1, seed)) >> 1) + ONE / 2;
            let dx = ox * ONE + jx - lx.frac;
            let dy = oy * ONE + jy - ly.frac;
            let d2 = mul(dx, dx) + mul(dy, dy);
            if d2 < best {
                best = d2;
            }
        }
    }
    fixed::sqrt(best)
}

/// Evaluate one basis.
#[inline]
pub fn basis(b: Basis, lx: Lat, ly: Lat, seed: u32) -> Fx {
    match b {
        Basis::Value => value2(lx, ly, seed),
        Basis::Gradient => gradient2(lx, ly, seed),
        Basis::Worley => worley2(lx, ly, seed) * 2 - ONE,
    }
}

/// Per-octave seed step, matching the floating-point path.
const SEED_STEP: u32 = 0x9e37_79b1;

/// Fractal sum.
///
/// `recip` is the reciprocal of [`crate::fixed::fractal_norm`] for this
/// octave count, computed by the caller. The device is given the same
/// number, so neither side divides.
pub fn fbm(b: Basis, lx: Lat, ly: Lat, octaves: u32, seed: u32, recip: Fx) -> Fx {
    let mut sum = 0;
    let mut amp = ONE;
    let (mut cx, mut cy) = (lx, ly);

    for o in 0..octaves.max(1) {
        sum += mul(
            amp,
            basis(b, cx, cy, seed.wrapping_add(o.wrapping_mul(SEED_STEP))),
        );
        amp >>= 1;
        cx = cx.double();
        cy = cy.double();
    }
    mul(sum, recip)
}

/// Ridged fractal sum, sharpened per octave.
pub fn ridged(
    b: Basis,
    lx: Lat,
    ly: Lat,
    octaves: u32,
    sharpness: u32,
    seed: u32,
    recip: Fx,
) -> Fx {
    let mut sum = 0;
    let mut amp = ONE;
    let (mut cx, mut cy) = (lx, ly);

    for o in 0..octaves.max(1) {
        let v = basis(b, cx, cy, seed.wrapping_add(o.wrapping_mul(SEED_STEP)));
        let r = ONE - v.abs();
        // Integer exponent by repeated multiplication, as in the WGSL
        // mirror, where `pow` is not exactly specified.
        let mut sharp = r;
        for _ in 1..sharpness.max(1) {
            sharp = mul(sharp, r);
        }
        sum += mul(amp, sharp);
        amp >>= 1;
        cx = cx.double();
        cy = cy.double();
    }
    mul(sum, recip)
}

/// Domain-warped fractal sum.
///
/// Takes the whole [`Prepared`] rather than eight loose arguments: it
/// needs the basis, the octave count, the strength and both
/// reciprocals, and threading those through individually was neither
/// readable nor hard to get out of order.
pub fn warped(p: &Prepared, lx: Lat, ly: Lat) -> Fx {
    let wx = fbm(
        Basis::Value,
        lx.offset(WARP_X0),
        ly.offset(WARP_Y0),
        2,
        p.seed ^ 0x5f35_6495,
        p.recip2,
    );
    let wy = fbm(
        Basis::Value,
        lx.offset(WARP_X1),
        ly.offset(WARP_Y1),
        2,
        p.seed ^ 0x3c6e_f372,
        p.recip2,
    );
    fbm(
        p.basis,
        lx.offset(mul(p.warp, wx)),
        ly.offset(mul(p.warp, wy)),
        p.octaves,
        p.seed,
        p.recip,
    )
}

// The warp offsets. Named constants rather than literals at the call
// site so that the shader can carry the same four integers and a test
// can check that it does.
const WARP_X0: Fx = 697_932_160; // 5.2
const WARP_Y0: Fx = 174_483_040; // 1.3
const WARP_X1: Fx = 228_170_144; // 1.7
const WARP_Y1: Fx = 1_234_803_072; // 9.2

/// The four warp offsets, for the shader to be checked against.
pub const WARP_OFFSETS: [Fx; 4] = [WARP_X0, WARP_Y0, WARP_X1, WARP_Y1];

/// Everything the synthesis needs that the host computes once.
///
/// Bundling it makes the device parameter block and the host renderer
/// share one definition, which is the only way the two stay in step
/// when a knob is added.
#[derive(Debug, Clone, Copy)]
pub struct Prepared {
    /// Which primitive the fractal sum is built from.
    pub basis: Basis,
    /// How the octaves are combined.
    pub pattern: Pattern,
    /// Octave count, at least one.
    pub octaves: u32,
    /// Ridge exponent, at least one.
    pub sharpness: u32,
    /// Hash seed.
    pub seed: u32,
    /// Domain warp strength.
    pub warp: Fx,
    /// Contrast multiplier about `pivot`.
    pub contrast: Fx,
    /// The value contrast pivots around.
    pub pivot: Fx,
    /// Reciprocal of the normalisation for `octaves` octaves.
    pub recip: Fx,
    /// Reciprocal of the normalisation for the two-octave warp fields.
    pub recip2: Fx,
}

impl Prepared {
    /// Convert a material into the form both paths consume.
    pub fn new(m: &Material) -> Self {
        Self {
            basis: m.basis,
            pattern: m.pattern,
            octaves: m.octaves.max(1),
            sharpness: m.sharpness.max(1),
            seed: m.seed,
            warp: fixed::from_f32(m.warp),
            contrast: fixed::from_f32(m.contrast),
            pivot: fixed::from_f32(m.pivot),
            recip: fixed::recip(fixed::fractal_norm(m.octaves)),
            recip2: fixed::recip(fixed::fractal_norm(2)),
        }
    }
}

/// Evaluate a prepared material at one lattice coordinate.
pub fn sample(p: &Prepared, lx: Lat, ly: Lat) -> f32 {
    let raw = match p.pattern {
        Pattern::Fractal => fbm(p.basis, lx, ly, p.octaves, p.seed, p.recip),
        Pattern::Ridged => ridged(p.basis, lx, ly, p.octaves, p.sharpness, p.seed, p.recip),
        Pattern::Warped => warped(p, lx, ly),
    };

    // Only the signed patterns are remapped; `ridged` already returns
    // [0, 1] and remapping again would compress it into [0.5, 1].
    let unit = match p.pattern {
        Pattern::Ridged => raw,
        _ => (raw >> 1) + ONE / 2,
    };

    let v = mul(unit - p.pivot, p.contrast) + p.pivot;
    fixed::to_f32(v.clamp(0, ONE))
}

impl Material {
    /// Render one tile with the fixed-point path.
    ///
    /// Produces the same picture [`Material::render_tile`] produces, to
    /// within the difference between 27 fractional bits and 24
    /// significant ones, and produces it identically on every GPU driver
    /// rather than only on ones that decline to fuse a multiply and an
    /// add.
    pub fn render_tile_exact(&self, tile: TileId, size: u32) -> Tile {
        let p = Prepared::new(self);
        let freq = self.frequency as f64;
        let ox = Lat::split(tile.x as f64 * freq);
        let oy = Lat::split(tile.y as f64 * freq);
        let step = fixed::from_f32(self.frequency / size as f32);

        let mut data = Vec::with_capacity((size * size) as usize);
        for y in 0..size {
            let ly = oy.step(y, step);
            for x in 0..size {
                data.push(sample(&p, ox.step(x, step), ly));
            }
        }
        Tile { size, data }
    }

    /// Render the unit tile at the origin with the fixed-point path.
    pub fn render_exact(&self, size: u32) -> Tile {
        self.render_tile_exact(TileId::new(0, 0), size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn material() -> Material {
        Material {
            basis: Basis::Gradient,
            pattern: Pattern::Ridged,
            frequency: 6.0,
            octaves: 5,
            sharpness: 3,
            contrast: 1.4,
            ..Default::default()
        }
    }

    #[test]
    fn the_warp_constants_are_the_conversions_they_claim_to_be() {
        // Written out so the shader can carry the same four integers.
        // Deriving them by hand from the decimal literals gave four
        // wrong numbers, because 5.2f32 is not 5.2; they come from the
        // f32 values that are actually converted.
        for (got, from) in WARP_OFFSETS.iter().zip([5.2f32, 1.3, 1.7, 9.2]) {
            assert_eq!(*got, fixed::from_f32(from), "constant for {from}");
        }
    }

    #[test]
    fn output_stays_in_range() {
        for basis in [Basis::Value, Basis::Gradient, Basis::Worley] {
            for pattern in [Pattern::Fractal, Pattern::Ridged, Pattern::Warped] {
                let m = Material {
                    basis,
                    pattern,
                    warp: 0.8,
                    ..material()
                };
                let t = m.render_exact(32);
                assert!(
                    t.data.iter().all(|v| (0.0..=1.0).contains(v)),
                    "{basis:?}/{pattern:?} left [0, 1]"
                );
            }
        }
    }

    #[test]
    fn it_produces_the_same_picture_as_the_floating_point_path() {
        // Not the same bits: 27 fractional bits and 24 significant ones
        // are different formats and round differently. The claim is that
        // the material is the same material, so compare after the 8-bit
        // quantisation the output actually goes through.
        for basis in [Basis::Value, Basis::Gradient, Basis::Worley] {
            for pattern in [Pattern::Fractal, Pattern::Ridged, Pattern::Warped] {
                let m = Material {
                    basis,
                    pattern,
                    warp: 0.8,
                    ..material()
                };
                let (a, b) = (m.render(64), m.render_exact(64));
                let differing = a
                    .data
                    .iter()
                    .zip(&b.data)
                    .filter(|(x, y)| {
                        crate::png::quantise(**x).abs_diff(crate::png::quantise(**y)) > 1
                    })
                    .count();
                let budget = a.data.len() / 100;
                assert!(
                    differing <= budget,
                    "{basis:?}/{pattern:?}: {differing} of {} pixels differ by more \
                     than one 8-bit level, above the budget of {budget}",
                    a.data.len()
                );
            }
        }
    }

    #[test]
    fn far_tiles_keep_their_detail() {
        let m = material();
        let baseline = m
            .render_tile_exact(TileId::new(0, 0), 24)
            .data
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        for &t in &[1i64 << 10, 1 << 20, m.max_tile()] {
            let distinct = m
                .render_tile_exact(TileId::new(t, 0), 24)
                .data
                .iter()
                .map(|v| v.to_bits())
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            assert!(
                distinct * 4 >= baseline * 3,
                "tile {t} resolves {distinct} values against {baseline} at tile 0"
            );
        }
    }

    #[test]
    fn tiles_meet_without_a_seam() {
        // The right edge of tile 0 and the left edge of tile 1 are
        // different evaluations of the same field, one cell apart, so
        // they should differ by no more than neighbouring pixels within
        // a tile do.
        let m = material();
        let size = 64;
        let a = m.render_tile_exact(TileId::new(0, 0), size);
        let b = m.render_tile_exact(TileId::new(1, 0), size);

        let across: f32 = (0..size)
            .map(|y| (a.get(size - 1, y) - b.get(0, y)).abs())
            .fold(0.0, f32::max);
        let within: f32 = (0..size)
            .map(|y| (a.get(size - 2, y) - a.get(size - 1, y)).abs())
            .fold(0.0, f32::max);
        assert!(
            across <= within * 3.0,
            "seam: {across} across the join against {within} within the tile"
        );
    }

    #[test]
    fn the_lattice_step_survives_a_wide_tile() {
        // `step` walks the tile by multiplying the pixel index by the
        // step size. That product exceeds 32 bits once the tile is wide
        // and the frequency is high, which is why it is taken at full
        // width; this is the case that would overflow if it were not.
        let m = Material {
            frequency: 15.0,
            ..material()
        };
        let t = m.render_exact(512);
        let distinct = t
            .data
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(
            distinct > 40_000,
            "a 512-wide tile at frequency 15 resolved only {distinct} values"
        );
    }
}
