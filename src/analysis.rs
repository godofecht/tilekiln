//! Parameter stability analysis.
//!
//! Everything above this module generates textures. This module answers
//! a different question: *which of your knobs is load-bearing, and
//! where does that stop being true?*
//!
//! # The construction
//!
//! A [`Material`] plus a knob index is the base state. The perturbation
//! jitters that one knob by a relative intensity drawn from
//! `U[0, max]`. The forward model renders a tile and reduces it to a
//! [`Feature`]. The invariance functional is the dispersion of that
//! feature across the ensemble.
//!
//! So the reported number is: *how far does this measurement move when
//! a designer nudges this knob by up to a few percent?* Large means the
//! knob is dangerous at this operating point. Small means it is safe to
//! expose to an artist.
//!
//! # Why one knob at a time
//!
//! Perturbing every knob together measures the aggregate, which is
//! dominated by whichever knob happens to have the largest effect and
//! tells you nothing about the others. Holding the rest fixed
//! attributes the movement, which is the whole point.
//!
//! # What the engine contributes
//!
//! Determinism. Draw `i` reads the substream `fork(seed, i)`, so the
//! ensemble is order-free and the analysis is reproducible bit for bit
//! across machines, thread counts and CPU vector widths. An analysis
//! whose answer depended on the core count would not be evidence of
//! anything.

use perturbation_kernel::config::{Accuracy, Backend, Config, Lipschitz};
use perturbation_kernel::engine::Engine;
use perturbation_kernel::forward::ForwardModel;
use perturbation_kernel::invariance::Invariance;
use perturbation_kernel::perturbation::Perturbation;
use perturbation_kernel::report::Report;
use perturbation_kernel::{reduce, Rng};

use crate::features::Feature;
use crate::material::Material;

/// How much a knob moves a measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Sensitivity {
    /// Which knob was perturbed.
    pub knob: &'static str,
    /// Root-mean-square movement of the feature under perturbation, in
    /// the feature's own units.
    pub spread: f64,
    /// The feature's value at the unperturbed material, for scale.
    pub nominal: f64,
    /// Theorem 7.3 error bound on `spread`, when the feature declares a
    /// Wasserstein-1 Lipschitz constant. `None` otherwise; see
    /// [`crate::features`].
    pub error_bound: Option<f64>,
}

impl Sensitivity {
    /// Movement relative to the nominal value.
    ///
    /// `None` when the nominal value is ~0, where a ratio is not
    /// meaningful rather than merely large.
    pub fn relative(&self) -> Option<f64> {
        if self.nominal.abs() < 1e-9 {
            None
        } else {
            Some(self.spread / self.nominal.abs())
        }
    }
}

/// Perturb exactly one knob by a relative amount.
struct KnobJitter {
    knob: usize,
    max_relative: f64,
}

impl Perturbation<Material> for KnobJitter {
    type Theta = f64;

    fn null(&self) -> f64 {
        0.0
    }

    fn sample_theta(&self, rng: &mut Rng) -> f64 {
        use rand::Rng as _;
        // Uniform on [0, max]: the intensity law rho. Drawn through the
        // engine's stream so the draw is addressable by index.
        rng.gen::<f64>() * self.max_relative
    }

    fn apply(&self, s: &Material, theta: &f64, rng: &mut Rng) -> Material {
        use rand::Rng as _;
        // Symmetric relative jitter. A knob at zero would be immovable
        // under a purely multiplicative nudge, so fall back to an
        // absolute one there.
        let d = (rng.gen::<f64>() * 2.0 - 1.0) * theta;
        let current = s.knob(self.knob).unwrap_or(0.0) as f64;
        let target = if current.abs() < 1e-9 {
            d
        } else {
            current * (1.0 + d)
        };

        // A quantised knob needs stochastic rounding, not nearest.
        // Nearest would swallow every nudge smaller than half a step
        // and report the knob as perfectly stable; stochastic rounding
        // crosses the step with probability equal to the fractional
        // part, so a 5% nudge on `octaves = 4` moves it 5% of the time
        // and the measured spread is the honest one.
        let value = if Material::KNOB_QUANTISED[self.knob] {
            let floor = target.floor();
            let frac = target - floor;
            if rng.gen::<f64>() < frac {
                floor + 1.0
            } else {
                floor
            }
        } else {
            target
        };
        s.with_knob(self.knob, value as f32)
    }
}

/// Render the material and reduce it to one number.
struct RenderAndMeasure {
    feature: Feature,
    size: u32,
}

impl ForwardModel<Material, f64> for RenderAndMeasure {
    fn eval(&self, m: &Material) -> f64 {
        self.feature.measure(&m.render(self.size)) as f64
    }
}

/// Dispersion of the measured feature. Reported as the negative
/// variance, matching the schema's convention that larger means more
/// invariant.
struct Dispersion {
    feature: Feature,
}

impl Invariance<f64> for Dispersion {
    fn measure(&self, ensemble: &[f64]) -> Report {
        let m = reduce::mean(ensemble);
        let var = reduce::sum_sq_dev(ensemble, m) / ensemble.len() as f64;
        Report::raw(
            -var,
            self.feature.name(),
            ensemble.len() as u64,
            0,
            Default::default(),
        )
    }

    fn lipschitz_w1(&self) -> Option<f64> {
        self.feature.lipschitz_w1()
    }

    fn name(&self) -> &str {
        self.feature.name()
    }
}

/// How to run an analysis.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// Perturbed renders per knob. The estimator's standard error falls
    /// as `1/sqrt(n)`, so 4096 buys about 1.5%.
    pub samples: u64,
    /// Tile resolution for each render. The dominant cost: total work
    /// is `samples × size²`.
    pub size: u32,
    /// Maximum relative jitter. 0.05 is "an artist nudged the slider".
    pub max_relative: f64,
    /// Which measurement to hold still.
    pub feature: Feature,
    /// Seed for the perturbation ensemble. Fixes the whole analysis.
    pub seed: u64,
    /// Execution backend. All host backends agree bit for bit.
    pub backend: Backend,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            samples: 4_096,
            size: 48,
            max_relative: 0.05,
            feature: Feature::Mean,
            seed: 0x07e5_5e4a,
            backend: Backend::Auto,
        }
    }
}

impl Settings {
    fn config(&self, feature: Feature) -> Config {
        let mut cfg = Config {
            n: self.samples,
            seed: self.seed,
            backend: self.backend,
            lipschitz: Lipschitz {
                forward_l: None,
                invariance_lambda: feature.lipschitz_w1(),
            },
            ..Default::default()
        };
        // Only claim an accuracy target the feature can actually
        // support. Asking for one without a Lipschitz constant would
        // produce a bound with nothing behind it.
        if let (Some(_), Some(d)) = (feature.lipschitz_w1(), feature.observation_diameter()) {
            cfg.accuracy = Some(Accuracy {
                // Loose enough that the sample floor does not reject
                // ordinary settings; the reported bound is what matters.
                epsilon: 1.0,
                eta: 0.05,
                observation_diameter: d,
                obs_dim: 1,
            });
        }
        cfg
    }
}

/// Measure how much one knob moves the chosen feature.
pub fn sensitivity(base: &Material, knob: usize, s: &Settings) -> Option<Sensitivity> {
    let name = *Material::KNOBS.get(knob)?;
    let model = RenderAndMeasure {
        feature: s.feature,
        size: s.size,
    };
    let nominal = model.eval(base);

    let report = Engine::run(
        base,
        &KnobJitter {
            knob,
            max_relative: s.max_relative,
        },
        &model,
        &Dispersion { feature: s.feature },
        &s.config(s.feature),
    )
    .ok()?;

    Some(Sensitivity {
        knob: name,
        // The report carries negative variance; the interpretable
        // quantity is its root, in the feature's units.
        spread: (-report.value).max(0.0).sqrt(),
        nominal,
        error_bound: report
            .error_bound
            .available
            .then_some(report.error_bound.epsilon),
    })
}

/// Measure every knob, most sensitive first.
///
/// The ordering is the deliverable: it says which slider to guard and
/// which to expose. It is *not* stable across parameter space, which is
/// the finding that motivates measuring rather than assuming.
pub fn sensitivities(base: &Material, s: &Settings) -> Vec<Sensitivity> {
    let mut out: Vec<_> = (0..Material::KNOBS.len())
        .filter_map(|i| sensitivity(base, i, s))
        .collect();
    out.sort_by(|a, b| {
        b.spread
            .partial_cmp(&a.spread)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Walk a knob across a range and report where the material stops being
/// stable.
///
/// Returns `(value, spread)` pairs. A sharp rise marks a boundary a
/// parameter preset should stay clear of.
pub fn sweep(
    base: &Material,
    knob: usize,
    from: f32,
    to: f32,
    steps: usize,
    s: &Settings,
) -> Vec<(f32, f64)> {
    (0..steps.max(2))
        .filter_map(|i| {
            let t = i as f32 / (steps.max(2) - 1) as f32;
            let v = from + (to - from) * t;
            let m = base.with_knob(knob, v);
            sensitivity(&m, knob, s).map(|r| (v, r.spread))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Pattern;
    use crate::noise::Basis;

    fn quick() -> Settings {
        Settings {
            samples: 256,
            size: 24,
            ..Default::default()
        }
    }

    #[test]
    fn a_null_perturbation_moves_nothing() {
        // C2: at zero intensity the material is returned unchanged, so
        // the ensemble is constant and the dispersion is exactly zero.
        let s = Settings {
            max_relative: 0.0,
            ..quick()
        };
        let r = sensitivity(&Material::default(), 0, &s).unwrap();
        assert_eq!(r.spread, 0.0);
    }

    #[test]
    fn the_analysis_is_reproducible() {
        let m = Material::default();
        let a = sensitivity(&m, 0, &quick()).unwrap();
        let b = sensitivity(&m, 0, &quick()).unwrap();
        assert_eq!(a.spread.to_bits(), b.spread.to_bits());
    }

    #[test]
    fn backends_agree_bit_for_bit() {
        // Inherited from the engine, and worth pinning here: an
        // analysis whose answer depended on the core count would not be
        // evidence of anything.
        let m = Material::default();
        let scalar = sensitivity(
            &m,
            0,
            &Settings {
                backend: Backend::Scalar,
                ..quick()
            },
        )
        .unwrap();
        let auto = sensitivity(
            &m,
            0,
            &Settings {
                backend: Backend::Auto,
                ..quick()
            },
        )
        .unwrap();
        assert_eq!(scalar.spread.to_bits(), auto.spread.to_bits());
    }

    #[test]
    fn a_bigger_nudge_moves_things_more() {
        let m = Material {
            pattern: Pattern::Ridged,
            basis: Basis::Gradient,
            ..Default::default()
        };
        let small = sensitivity(
            &m,
            0,
            &Settings {
                max_relative: 0.01,
                ..quick()
            },
        )
        .unwrap();
        let large = sensitivity(
            &m,
            0,
            &Settings {
                max_relative: 0.20,
                ..quick()
            },
        )
        .unwrap();
        assert!(
            large.spread > small.spread,
            "{} vs {}",
            small.spread,
            large.spread
        );
    }

    #[test]
    fn mean_carries_an_error_bound_and_the_others_do_not() {
        let m = Material::default();
        let with = sensitivity(
            &m,
            0,
            &Settings {
                feature: Feature::Mean,
                ..quick()
            },
        )
        .unwrap();
        assert!(with.error_bound.is_some());
        let without = sensitivity(
            &m,
            0,
            &Settings {
                feature: Feature::EdgeDensity,
                ..quick()
            },
        )
        .unwrap();
        assert!(without.error_bound.is_none());
    }

    #[test]
    fn quantised_knobs_actually_move() {
        // The bug this guards: nearest-rounding a 5% nudge on
        // `octaves = 4` never leaves 4, so the knob reported a spread
        // of exactly zero and looked perfectly stable.
        let m = Material {
            octaves: 4,
            ..Default::default()
        };
        let r = sensitivity(
            &m,
            1,
            &Settings {
                feature: Feature::EdgeDensity,
                size: 32,
                ..quick()
            },
        )
        .unwrap();
        assert!(r.spread > 0.0, "the octaves knob did not move at all");
    }

    #[test]
    fn an_unknown_knob_is_none() {
        assert!(sensitivity(&Material::default(), 99, &quick()).is_none());
    }

    #[test]
    fn sensitivities_come_back_sorted() {
        let all = sensitivities(&Material::default(), &quick());
        assert_eq!(all.len(), Material::KNOBS.len());
        for w in all.windows(2) {
            assert!(w[0].spread >= w[1].spread);
        }
    }
}
