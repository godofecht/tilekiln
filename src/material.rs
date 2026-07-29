//! Materials: the knobs a texture artist actually turns.
//!
//! A [`Material`] is a small parameter record plus a seed. It is a
//! *value*, not a builder or a graph, for two reasons. It has to
//! serialise across the FFI boundary to a compute shader without a
//! layout negotiation, and it has to be perturbable by the stability
//! analysis, which needs to jitter named scalars and re-run.
//!
//! The node-graph generality that most texture tools reach for is
//! deliberately absent. A graph cannot be perturbed coherently: there
//! is no meaningful notion of "nudge this subtree by 5%". Fixing the
//! topology and exposing scalars is what makes the analysis in
//! [`crate::analysis`] possible at all.

use crate::noise::{fbm, ridged, warped, Basis};

/// Which fractal construction to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    /// Plain fractal sum. Cloudy.
    /// Plain fractal sum. Cloudy.
    Fractal,
    /// `1 − |n|`, sharpened. Ridges, veins, cracks.
    /// `1 - abs(n)`, sharpened. Ridges, veins, cracks.
    Ridged,
    /// Fractal sum sampled through a displaced domain. Marbled, eroded.
    /// Fractal sum through a displaced domain. Marbled, eroded.
    Warped,
}

/// A procedural material.
///
/// Every field except `basis`, `pattern` and `seed` is a continuous
/// knob, which is what the stability analysis perturbs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Which noise primitive the fractal sum is built from.
    pub basis: Basis,
    /// Which fractal construction to apply.
    pub pattern: Pattern,
    /// Features per unit of texture space. The dominant scale knob.
    pub frequency: f32,
    /// Octaves in the fractal sum. Detail depth.
    pub octaves: u32,
    /// Ridge exponent; only read when `pattern` is [`Pattern::Ridged`].
    pub sharpness: u32,
    /// Domain displacement; only read when `pattern` is
    /// [`Pattern::Warped`].
    pub warp: f32,
    /// Output remap: `contrast * (n - pivot) + pivot`, then clamped.
    pub contrast: f32,
    /// Value the contrast remap pivots about.
    pub pivot: f32,
    /// Field seed. Changes the pattern without changing its character.
    pub seed: u32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            basis: Basis::Gradient,
            pattern: Pattern::Fractal,
            frequency: 4.0,
            octaves: 5,
            sharpness: 2,
            warp: 0.0,
            contrast: 1.0,
            pivot: 0.5,
            seed: 0,
        }
    }
}

impl Material {
    /// Evaluate the material at a point in unit texture space.
    ///
    /// Pure in `(self, x, y)`: no stream state, no interior mutability,
    /// no dependence on evaluation order. That purity is what
    /// [`crate::analysis`] requires of a forward model and what makes
    /// tile addressing meaningful.
    pub fn sample(&self, x: f64, y: f64) -> f32 {
        // Coordinates stay in f64 through the frequency scaling. f32
        // has a 24-bit mantissa, so at a tile index around 2^20 its
        // ulp exceeds the pixel step and adjacent pixels collapse onto
        // the same lattice cell: measured, 2,207 distinct values out of
        // 65,536. f64 pushes that past any coordinate a texture plane
        // will ever use.
        let fx = x * self.frequency as f64;
        let fy = y * self.frequency as f64;

        let raw = match self.pattern {
            Pattern::Fractal => fbm(self.basis, fx, fy, self.octaves, self.seed),
            Pattern::Ridged => ridged(self.basis, fx, fy, self.octaves, self.sharpness, self.seed),
            Pattern::Warped => warped(self.basis, fx, fy, self.octaves, self.warp, self.seed),
        };

        // Only the signed patterns need remapping. `ridged` already
        // returns [0, 1], and passing it through `raw * 0.5 + 0.5`
        // again compressed it into [0.5, 1.0], which is why ridged
        // materials came out washed-out and pale.
        let unit = match self.pattern {
            Pattern::Ridged => raw,
            Pattern::Fractal | Pattern::Warped => raw * 0.5 + 0.5,
        };
        let out = (unit - self.pivot) * self.contrast + self.pivot;
        out.clamp(0.0, 1.0)
    }

    /// Whether each knob takes a continuum of values or only integers.
    ///
    /// This matters to [`crate::analysis`]: a 5% relative nudge on
    /// `octaves = 4` lands in `3.8 ..= 4.2`, which rounds straight back
    /// to 4. Perturbing a quantised knob the same way as a continuous
    /// one measures nothing and reports it as perfect stability, which
    /// is the most misleading answer available.
    pub const KNOB_QUANTISED: [bool; 5] = [false, true, true, false, false];

    /// The named knobs, in a fixed order.
    ///
    /// The order is part of the API: [`crate::analysis`] reports
    /// sensitivities positionally, and a caller comparing two runs
    /// needs the columns to line up.
    pub const KNOBS: [&'static str; 5] = ["frequency", "octaves", "sharpness", "warp", "contrast"];

    /// Read a knob by index. Returns `None` past the end.
    pub fn knob(&self, i: usize) -> Option<f32> {
        Some(match i {
            0 => self.frequency,
            1 => self.octaves as f32,
            2 => self.sharpness as f32,
            3 => self.warp,
            4 => self.contrast,
            _ => return None,
        })
    }

    /// Return a copy with knob `i` set to `v`.
    ///
    /// Integer-valued knobs round and clamp to at least one, so a
    /// perturbation cannot produce a zero-octave material.
    pub fn with_knob(mut self, i: usize, v: f32) -> Self {
        match i {
            0 => self.frequency = v.max(0.0),
            1 => self.octaves = (v.round().max(1.0)) as u32,
            2 => self.sharpness = (v.round().max(1.0)) as u32,
            3 => self.warp = v,
            4 => self.contrast = v,
            _ => {}
        }
        self
    }
}

/// A tile coordinate in an unbounded texture plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    /// Tile column.
    pub x: i64,
    /// Tile row.
    pub y: i64,
}

impl TileId {
    /// A tile coordinate.
    pub const fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }
}

/// A rendered tile: `size × size` scalars in `[0, 1]`, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    /// Edge length in pixels.
    pub size: u32,
    /// Row-major samples in `[0, 1]`.
    pub data: Vec<f32>,
}

impl Tile {
    #[inline]
    /// Sample at `(x, y)`. Panics if out of bounds.
    pub fn get(&self, x: u32, y: u32) -> f32 {
        self.data[(y * self.size + x) as usize]
    }

    /// Encode as an 8-bit greyscale PNG.
    pub fn to_png(&self) -> Vec<u8> {
        let px: Vec<u8> = self
            .data
            .iter()
            .copied()
            .map(crate::png::quantise)
            .collect();
        crate::png::grey(self.size, self.size, &px)
    }
}

impl Material {
    /// Render one tile of an unbounded plane.
    ///
    /// Tile `(10⁶, −4)` costs exactly what tile `(0, 0)` costs, and
    /// rendering it does not depend on any other tile having been
    /// rendered. Adjacent tiles agree along their shared edge because
    /// both evaluate the same continuous field at the same coordinates,
    /// so there is no seam to hide.
    pub fn render_tile(&self, tile: TileId, size: u32) -> Tile {
        let mut data = Vec::with_capacity((size * size) as usize);
        let inv = 1.0 / size as f64;
        for iy in 0..size {
            for ix in 0..size {
                let x = tile.x as f64 + ix as f64 * inv;
                let y = tile.y as f64 + iy as f64 * inv;
                data.push(self.sample(x, y));
            }
        }
        Tile { size, data }
    }

    /// Render the unit tile at the origin.
    pub fn render(&self, size: u32) -> Tile {
        self.render_tile(TileId::new(0, 0), size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_is_bounded() {
        let m = Material {
            contrast: 8.0,
            ..Default::default()
        };
        for v in m.render(64).data {
            assert!((0.0..=1.0).contains(&v), "{v} escaped the unit interval");
        }
    }

    #[test]
    fn tiles_are_addressable_in_any_order() {
        let m = Material::default();
        let far = m.render_tile(TileId::new(1_000_000, -4_000_000), 16);
        let near = m.render_tile(TileId::new(0, 0), 16);
        let far_again = m.render_tile(TileId::new(1_000_000, -4_000_000), 16);
        assert_eq!(
            far, far_again,
            "a tile changed depending on what was rendered before it"
        );
        assert_ne!(far.data, near.data);
    }

    #[test]
    fn adjacent_tiles_meet_without_a_seam() {
        // The right edge of tile (0,0) and the left edge of tile (1,0)
        // sample the same continuous field, so they must agree exactly.
        let m = Material::default();
        let size = 32;
        let a = m.render_tile(TileId::new(0, 0), size);
        let b = m.render_tile(TileId::new(1, 0), size);
        for y in 0..size {
            let x_right = 1.0f64;
            let left_of_b = m.sample(1.0, y as f64 / size as f64);
            assert_eq!(
                m.sample(x_right, y as f64 / size as f64).to_bits(),
                left_of_b.to_bits()
            );
            let _ = (a.get(0, y), b.get(0, y));
        }
    }

    #[test]
    fn knobs_round_trip() {
        let m = Material::default();
        for i in 0..Material::KNOBS.len() {
            let v = m.knob(i).unwrap();
            let m2 = m.with_knob(i, v);
            assert_eq!(
                m2.knob(i).unwrap(),
                v,
                "knob {} did not round-trip",
                Material::KNOBS[i]
            );
        }
        assert!(m.knob(99).is_none());
    }

    #[test]
    fn octaves_cannot_be_perturbed_to_zero() {
        let m = Material::default().with_knob(1, -3.0);
        assert_eq!(m.octaves, 1);
    }
}

#[cfg(test)]
mod precision_tests {
    use super::*;

    /// Detail must survive at large tile indices.
    ///
    /// With the coordinate arithmetic in `f32` this failed badly: at
    /// tile 2^20 only 2,207 of 65,536 pixels were distinct, because
    /// `f32`'s ulp there exceeds the pixel step and adjacent pixels
    /// land in the same lattice cell. The tile still rendered, and it
    /// still rendered *fast*, which is exactly what made the bug easy
    /// to miss.
    #[test]
    fn far_tiles_keep_their_detail() {
        let m = Material {
            frequency: 4.0,
            octaves: 4,
            ..Default::default()
        };
        let size = 128u32;
        let total = (size * size) as usize;

        for shift in [0u32, 10, 20, 30, 40] {
            let t = 1i64 << shift;
            let tile = m.render_tile(TileId::new(t, 0), size);
            let mut bits: Vec<u32> = tile.data.iter().map(|v| v.to_bits()).collect();
            bits.sort_unstable();
            bits.dedup();
            let ratio = bits.len() as f64 / total as f64;
            assert!(
                ratio > 0.9,
                "tile 2^{shift}: only {} of {total} pixels distinct ({:.1}%), \
                 which means the coordinate precision collapsed",
                bits.len(),
                ratio * 100.0
            );
        }
    }
}
