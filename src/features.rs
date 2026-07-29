//! Scalar features of a rendered tile.
//!
//! The stability analysis needs a number, not an image. Which number
//! decides what "the texture changed" means, so the choice is part of
//! the question rather than an implementation detail:
//!
//! * [`mean`] moves when the material gets lighter or darker overall.
//! * [`contrast`] moves when it gets flatter or punchier.
//! * [`edge_density`] moves when the *amount* of visible structure
//!   changes.
//! * [`spectral_centroid`] moves when the structure changes *scale*,
//!   which is the one a human calls "it looks like a different
//!   material" even at identical mean and contrast.
//!
//! # On Lipschitz constants
//!
//! `perturbation-kernel` derives a non-asymptotic error bound only for
//! functionals with a declared Wasserstein-1 Lipschitz constant. Tile
//! values live in `[0, 1]`, so the observation diameter is 1 and
//! [`mean`] is exactly 1-Lipschitz in `W₁`: it is an average of a
//! 1-Lipschitz function of the pixel distribution.
//!
//! # Resolution is part of the measurement
//!
//! [`edge_density`] and [`spectral_centroid`] read pixels, so they can
//! only see structure the tile resolves. An octave at `frequency ×
//! 2^(o-1)` cycles is invisible on a tile of fewer than twice that many
//! pixels, and adding such octaves *lowers* measured edge density,
//! because the fractal sum normalises by accumulated amplitude while
//! contributing nothing above Nyquist.
//!
//! Measured at frequency 6 on a 64-pixel tile:
//!
//! ```text
//! octaves      1       2       4       6       8
//! edges   0.0483  0.0430  0.0434  0.0420  0.0415   <- all past Nyquist
//! ```
//!
//! and at frequency 1 on a 256-pixel tile, where the octaves resolve:
//!
//! ```text
//! octaves      1       2       4       6       8
//! edges   0.0018  0.0016  0.0020  0.0023  0.0026   <- rises as intended
//! ```
//!
//! So an analysis run at too low a tile size will report the octaves
//! knob as *stabilising*, which is an artefact of the sampling rather
//! than a property of the material. Size the tile against the highest
//! frequency you care about.
//!
//! The other three are not. [`contrast`] is quadratic in the values,
//! and both [`edge_density`] and [`spectral_centroid`] depend on pixel
//! *arrangement* rather than on the pixel distribution, so they are not
//! functionals of the empirical measure at all. Each reports `None`
//! from [`Feature::lipschitz_w1`] and the analysis omits the error
//! bound instead of inventing a constant. That is the honest outcome,
//! and it is why [`Feature::Mean`] is the default.

use crate::material::Tile;

/// Mean pixel value.
pub fn mean(tile: &Tile) -> f32 {
    if tile.data.is_empty() {
        return 0.0;
    }
    let sum: f32 = tile.data.iter().sum();
    sum / tile.data.len() as f32
}

/// Root-mean-square deviation from the mean.
pub fn contrast(tile: &Tile) -> f32 {
    if tile.data.is_empty() {
        return 0.0;
    }
    let m = mean(tile);
    let ss: f32 = tile.data.iter().map(|v| (v - m) * (v - m)).sum();
    (ss / tile.data.len() as f32).sqrt()
}

/// Mean absolute finite-difference gradient magnitude.
///
/// A cheap proxy for "how much visible structure is there". Computed on
/// the interior only, so the tile edges do not contribute a spurious
/// step.
pub fn edge_density(tile: &Tile) -> f32 {
    let n = tile.size;
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let c = tile.get(x, y);
            let dx = tile.get(x + 1, y) - c;
            let dy = tile.get(x, y + 1) - c;
            sum += (dx * dx + dy * dy).sqrt();
            count += 1;
        }
    }
    sum / count as f32
}

/// Radially-averaged spectral centroid, in cycles per tile.
///
/// The centre of mass of the power spectrum: low when the texture is
/// dominated by large blobs, high when it is dominated by fine grain.
/// This is the feature that separates "same material, different
/// brightness" from "different material".
///
/// Uses a direct discrete transform over a bounded band rather than an
/// FFT. The band is capped at 32 cycles, which is where texture
/// character lives; going further costs quadratically and measures
/// mostly aliasing.
pub fn spectral_centroid(tile: &Tile) -> f32 {
    let n = tile.size;
    if n < 4 {
        return 0.0;
    }
    let m = mean(tile);
    let max_k = 32u32.min(n / 2);

    let mut num = 0.0f64;
    let mut den = 0.0f64;

    // Sample the spectrum along the two axes and the two diagonals.
    // A full 2D transform would be O(n⁴) here; four radial slices
    // capture the scale of the dominant structure at O(n² k).
    for k in 1..=max_k {
        let mut power = 0.0f64;
        const D: f32 = std::f32::consts::FRAC_1_SQRT_2;
        for &(ux, uy) in &[(1.0f32, 0.0f32), (0.0, 1.0), (D, D), (D, -D)] {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for y in 0..n {
                for x in 0..n {
                    let v = (tile.get(x, y) - m) as f64;
                    let proj = (x as f32 * ux + y as f32 * uy) as f64 / n as f64;
                    let ang = std::f64::consts::TAU * k as f64 * proj;
                    re += v * ang.cos();
                    im += v * ang.sin();
                }
            }
            power += re * re + im * im;
        }
        num += k as f64 * power;
        den += power;
    }

    if den <= 0.0 {
        0.0
    } else {
        (num / den) as f32
    }
}

/// Which scalar to measure a tile by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// Overall lightness. The only one with a declared `W₁` constant,
    /// so the only one that yields an error bound.
    Mean,
    /// Flatness against punchiness.
    Contrast,
    /// How much visible structure there is.
    EdgeDensity,
    /// What *scale* the structure is at.
    SpectralCentroid,
}

impl Feature {
    /// Measure `tile` with this feature.
    pub fn measure(self, tile: &Tile) -> f32 {
        match self {
            Feature::Mean => mean(tile),
            Feature::Contrast => contrast(tile),
            Feature::EdgeDensity => edge_density(tile),
            Feature::SpectralCentroid => spectral_centroid(tile),
        }
    }

    /// Stable tag, used as the report's functional name.
    pub fn name(self) -> &'static str {
        match self {
            Feature::Mean => "mean",
            Feature::Contrast => "contrast",
            Feature::EdgeDensity => "edge_density",
            Feature::SpectralCentroid => "spectral_centroid",
        }
    }

    /// Declared Wasserstein-1 Lipschitz constant, where one exists.
    ///
    /// `None` means the analysis will not report an error bound, which
    /// is the correct outcome for a functional that is not `W₁`
    /// Lipschitz. See the module documentation.
    pub fn lipschitz_w1(self) -> Option<f64> {
        match self {
            Feature::Mean => Some(1.0),
            _ => None,
        }
    }

    /// Diameter of the observation space, where it is bounded.
    pub fn observation_diameter(self) -> Option<f64> {
        match self {
            Feature::Mean => Some(1.0),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{Material, Pattern};
    use crate::noise::Basis;

    fn flat(size: u32, v: f32) -> Tile {
        Tile {
            size,
            data: vec![v; (size * size) as usize],
        }
    }

    #[test]
    fn a_flat_tile_has_no_structure() {
        let t = flat(32, 0.5);
        assert_eq!(mean(&t), 0.5);
        assert_eq!(contrast(&t), 0.0);
        assert_eq!(edge_density(&t), 0.0);
        assert_eq!(spectral_centroid(&t), 0.0);
    }

    #[test]
    fn spectral_centroid_tracks_frequency() {
        // The claim that makes this feature worth its cost: raising the
        // material frequency must raise the measured centroid.
        let base = Material {
            basis: Basis::Gradient,
            pattern: Pattern::Fractal,
            octaves: 1,
            ..Default::default()
        };
        let low = spectral_centroid(
            &Material {
                frequency: 2.0,
                ..base
            }
            .render(64),
        );
        let high = spectral_centroid(
            &Material {
                frequency: 16.0,
                ..base
            }
            .render(64),
        );
        assert!(
            high > low,
            "centroid did not rise with frequency: {low} -> {high}"
        );
    }

    #[test]
    fn edge_density_tracks_resolvable_detail() {
        // Measured at a frequency whose octaves the tile can actually
        // resolve. Octave 1 is excluded because it is a special case:
        // the fractal sum normalises by accumulated amplitude, so the
        // step from one octave to two *lowers* contrast before the
        // added detail starts winning.
        let base = Material {
            pattern: Pattern::Fractal,
            frequency: 1.0,
            ..Default::default()
        };
        let coarse = edge_density(&Material { octaves: 2, ..base }.render(256));
        let fine = edge_density(&Material { octaves: 8, ..base }.render(256));
        assert!(
            fine > coarse,
            "resolvable octaves gave fewer edges: {coarse} -> {fine}"
        );
    }

    #[test]
    fn unresolvable_octaves_do_not_add_measurable_detail() {
        // The caveat in the module documentation, pinned. At frequency
        // 6 on a 64-pixel tile, octave 8 sits at 768 cycles: far past
        // Nyquist. Adding those octaves cannot raise edge density, and
        // the amplitude normalisation means it slightly lowers it.
        let base = Material {
            pattern: Pattern::Fractal,
            frequency: 6.0,
            ..Default::default()
        };
        let few = edge_density(&Material { octaves: 2, ..base }.render(64));
        let many = edge_density(&Material { octaves: 8, ..base }.render(64));
        assert!(
            many <= few * 1.05,
            "octaves past Nyquist should not add detail: {few} -> {many}"
        );
    }

    #[test]
    fn only_mean_claims_a_lipschitz_constant() {
        assert_eq!(Feature::Mean.lipschitz_w1(), Some(1.0));
        for f in [
            Feature::Contrast,
            Feature::EdgeDensity,
            Feature::SpectralCentroid,
        ] {
            assert!(
                f.lipschitz_w1().is_none(),
                "{} should not claim a constant",
                f.name()
            );
        }
    }
}
