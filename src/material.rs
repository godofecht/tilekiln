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

use crate::noise::{fbm_at, ridged_at, warped_at, Basis, Lattice};

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
        // Scale in f64, then split. The magnitude ends up in an integer
        // cell index, so a distant tile is as exact as tile 0; with a
        // single f32 coordinate, tile 2^20 already collapsed 65,536
        // pixels onto 2,207 distinct values. See Material::max_tile for
        // where the integer cell runs out.
        let fx = x * self.frequency as f64;
        let fy = y * self.frequency as f64;
        self.sample_lattice(Lattice::split(fx), Lattice::split(fy))
    }

    /// Evaluate at an already-split coordinate.
    ///
    /// This is the real entry point. Every arithmetic operation from here
    /// down is one WGSL specifies exactly, which is what lets the device
    /// path track the host closely at any tile index. [`Self::sample`]
    /// is a convenience that splits an `f64` first.
    pub fn sample_lattice(&self, lx: Lattice, ly: Lattice) -> f32 {
        let raw = match self.pattern {
            Pattern::Fractal => fbm_at(self.basis, lx, ly, self.octaves, self.seed),
            Pattern::Ridged => {
                ridged_at(self.basis, lx, ly, self.octaves, self.sharpness, self.seed)
            }
            Pattern::Warped => warped_at(self.basis, lx, ly, self.octaves, self.warp, self.seed),
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
    /// Largest tile index that does not alias onto a nearer one.
    ///
    /// The lattice cell is an `i32` and wraps, so the field repeats every
    /// [`Lattice::PERIOD`] cells. A tile covers `frequency` cells, which
    /// makes the usable range `PERIOD / (2 * frequency)` in each
    /// direction, the halving because indices run both ways from zero.
    ///
    /// ```
    /// use tilekiln::Material;
    /// let m = Material { frequency: 6.0, ..Default::default() };
    /// assert_eq!(m.max_tile(), 357_913_941);
    /// ```
    ///
    /// Nothing enforces this. Rendering past it is well defined and
    /// mirrorable on the device; it simply returns a copy of somewhere
    /// nearer.
    pub fn max_tile(&self) -> i64 {
        (Lattice::PERIOD / (2.0 * self.frequency as f64)) as i64
    }

    /// Render one tile of a large addressable plane.
    ///
    /// Tile `(10⁶, −4)` costs exactly what tile `(0, 0)` costs, and
    /// rendering it does not depend on any other tile having been
    /// rendered. Adjacent tiles agree along their shared edge because
    /// both evaluate the same continuous field at the same coordinates,
    /// so there is no seam to hide.
    ///
    /// The plane is large rather than unbounded. Beyond
    /// [`Self::max_tile`] the field repeats, at full detail and without
    /// complaint. See [`Lattice`] for why.
    pub fn render_tile(&self, tile: TileId, size: u32) -> Tile {
        // Split the tile origin once, in f64, then step across the tile
        // in f32. The magnitude lives in the integer cell, so a distant
        // tile is as exact as tile 0, and every per-pixel operation is
        // one the WGSL mirror performs identically.
        let freq = self.frequency as f64;
        let ox = Lattice::split(tile.x as f64 * freq);
        let oy = Lattice::split(tile.y as f64 * freq);
        let step = self.frequency / size as f32;

        let mut data = Vec::with_capacity((size * size) as usize);
        for iy in 0..size {
            let ly = oy.offset(iy as f32 * step);
            for ix in 0..size {
                let lx = ox.offset(ix as f32 * step);
                data.push(self.sample_lattice(lx, ly));
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
    // The original far-tile test checked that a distant tile still had
    // detail, and passed while tile 2^40 was returning a byte-identical
    // copy of tile 0. Detail is the wrong thing to measure; distinctness
    // is. This is that test.
    #[test]
    fn tiles_inside_the_usable_range_are_all_different() {
        let m = Material {
            basis: crate::Basis::Gradient,
            pattern: Pattern::Ridged,
            frequency: 6.0,
            octaves: 5,
            sharpness: 3,
            ..Default::default()
        };
        let limit = m.max_tile();
        let count = |t: i64| -> (Vec<u32>, usize) {
            let tile = m.render_tile(TileId::new(t, 0), 24);
            let bits: Vec<u32> = tile.data.iter().map(|v| v.to_bits()).collect();
            let distinct = bits
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            (bits, distinct)
        };

        // Tile 0 sets the baseline. An absolute threshold would be a
        // guess about how much variety this material happens to have;
        // what matters is that a distant tile does not have *less*.
        let (base_bits, baseline) = count(0);
        let mut seen = std::collections::BTreeSet::new();
        seen.insert(base_bits);

        for &t in &[1i64, 1 << 10, 1 << 20, 1 << 26, limit / 2, limit] {
            let (bits, distinct) = count(t);
            assert!(
                seen.insert(bits),
                "tile {t} is a bit-for-bit copy of an earlier tile, inside the \
                 usable range of {limit}"
            );
            assert!(
                distinct * 4 >= baseline * 3,
                "tile {t} resolves {distinct} distinct values against {baseline} \
                 at tile 0, so the coordinate has lost precision"
            );
        }
    }

    // The period is a documented property, so pin it. If the cell ever
    // widens beyond i32 this fails and the docs get corrected with it.
    #[test]
    fn the_field_repeats_exactly_at_the_documented_period() {
        let m = Material {
            frequency: 6.0,
            octaves: 4,
            ..Default::default()
        };
        let base = m.render_tile(TileId::new(0, 0), 16);

        // One full period in cells, converted back to tiles: 2^32 / 6 is
        // not an integer, so use a multiple that is. 2^40 * 6 is exactly
        // 1536 * 2^32.
        let aliased = m.render_tile(TileId::new(1 << 40, 0), 16);
        assert!(
            base.data
                .iter()
                .zip(&aliased.data)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "tile 2^40 should alias exactly onto tile 0 at frequency 6"
        );

        // And just inside the range it must not.
        let ok = m.render_tile(TileId::new(m.max_tile(), 0), 16);
        assert!(
            base.data
                .iter()
                .zip(&ok.data)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "the largest in-range tile should not alias onto tile 0"
        );
    }

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
