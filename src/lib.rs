//! Tile-addressable procedural texture synthesis with measured
//! parameter stability.
//!
//! Two things, deliberately in one crate because each is worth more
//! with the other.
//!
//! # Synthesis
//!
//! [`Material`] is a small record of knobs. Sampling it is a pure
//! function of position, with no stream state anywhere, so tile
//! (10⁶, −4) costs what tile (0, 0) costs and rendering it does not
//! depend on which tiles came before. Adjacent tiles meet without a
//! seam because both evaluate the same continuous field.
//!
//! ```
//! use tilekiln::{Material, TileId};
//!
//! let m = Material { frequency: 6.0, octaves: 5, ..Default::default() };
//! let tile = m.render_tile(TileId::new(1_000_000, -4), 256);
//! assert_eq!(tile.data.len(), 256 * 256);
//! ```
//!
//! Every operation in the synthesis path is `+`, `-` or `*` on `f32`,
//! plus one correctly-rounded `sqrt`. WGSL specifies those exactly and
//! leaves `sin`, `exp` and `pow` to the driver, so avoiding the latter
//! holds the GPU path to 9e-6 of the CPU path rather than merely
//! similar. See [`noise`].
//!
//! That is as close as floating point gets, because a shader compiler
//! may fuse a multiply and an add into one `fma` and WGSL cannot forbid
//! it: llvmpipe matches the host bit for bit, Metal does not. For
//! agreement that does not depend on the driver there is [`exact`],
//! which runs the same construction in Q4.27 integers and is
//! bit-identical everywhere, at 1.6x to 4.2x the cost.
//!
//! # Stability
//!
//! A texture tool will happily hand an artist a slider that reshuffles
//! the entire material on a 2% nudge, and nothing in the tool says so.
//! [`analysis`] measures it: perturb one knob, re-render, and report
//! how far a chosen [`features::Feature`] moved.
//!
//! ```no_run
//! use tilekiln::{analysis, Material};
//!
//! for s in analysis::sensitivities(&Material::default(), &Default::default()) {
//!     println!("{:<12} {:.5}", s.knob, s.spread);
//! }
//! ```
//!
//! The ranking is the deliverable, and it is not constant across
//! parameter space. A knob that is safe at one operating point can
//! dominate at another, which is why this is measured rather than
//! assumed.
//!
//! The estimator underneath is `perturbation-kernel`, so an analysis is
//! a pure function of `(material, knob, settings)` and reproduces bit
//! for bit across machines, thread counts and CPU vector widths. An
//! analysis whose answer depended on the core count would not be
//! evidence of anything.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod exact;
pub mod features;
pub mod fixed;
pub mod hash;
pub mod material;
pub mod noise;
pub mod png;

#[cfg(feature = "analysis")]
pub mod analysis;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use features::Feature;
pub use material::{Material, Pattern, Tile, TileId};
pub use noise::Basis;
