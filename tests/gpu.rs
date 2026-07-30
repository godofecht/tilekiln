//! How closely does the device match the host?
//!
//! The synthesis path was built for bit-identity: hashed gradients
//! instead of trigonometry, a quintic polynomial interpolant, repeated
//! multiplication instead of `pow`, lacunarity pinned at 2, coordinates
//! split into an integer cell and a fractional offset.
//!
//! Whether it gets there depends on the driver. A shader compiler may
//! fuse a multiply and an add into one `fma`, rounding once where the
//! host rounds twice, and WGSL cannot forbid it. On llvmpipe, 8 of the 9
//! basis/pattern combinations below come back bit-identical, so the
//! arithmetic is expressed exactly. On Apple's Metal compiler none of
//! them do, because nearly every line of the path is a multiply followed
//! by an add.
//!
//! So these tests pin a bound and report which combinations were exact,
//! rather than asserting equality and failing on half the machines that
//! run them.
//!
//! Units in the last place are the wrong metric for the bound: ulp shrink
//! towards zero, so a fixed absolute error near a dark pixel counts as
//! tens of thousands of them. Absolute error is what matters, because the
//! output is quantised to eight bits. One level is 3.9e-3; the worst
//! disagreement measured over 311,040 samples was 8.9e-6, some 440 times
//! smaller, and 5 of those samples sat close enough to a boundary to
//! round to a different byte.
//!
//! Exactness on *every* driver means not using floats at all, which is
//! what `Material::render_tile_exact` does: Q4.27 integers throughout,
//! with the 64-bit multiply and the integer square root spelled out in
//! `synth_exact.wgsl` because WGSL has no integer wider than 32 bits.
//! The tests at the bottom of this file assert `to_bits()` equality for
//! that path, with no tolerance, and it holds on Metal where the
//! floating-point path manages none of the nine combinations.

#![cfg(feature = "gpu")]

use tilekiln::gpu::Gpu;
use tilekiln::material::Pattern;
use tilekiln::{Basis, Material, TileId};

/// Acquire a device, or skip. `TILEKILN_REQUIRE_GPU=1` turns the skip
/// into a failure, so CI cannot pass vacuously on a runner where a
/// device was installed on purpose.
macro_rules! gpu_or_skip {
    () => {
        match Gpu::new() {
            Ok(g) => g,
            Err(e) => {
                if std::env::var_os("TILEKILN_REQUIRE_GPU").is_some() {
                    panic!("TILEKILN_REQUIRE_GPU is set but no device was found: {e}");
                }
                eprintln!("SKIP: {e}");
                return;
            }
        }
    };
}

/// The bound the floating-point path actually holds to.
///
/// Measured, not chosen for comfort: the observed worst case is 8.9e-6,
/// so this leaves a factor of two and no more. It is still 175 times
/// below one 8-bit quantisation level.
const MAX_ABS_ERROR: f32 = 2.0e-5;

/// One level of the 8-bit output the renders are quantised to.
const QUANT_STEP: f32 = 1.0 / 255.0;

fn compare(gpu: &Gpu, m: &Material, tile: TileId, size: u32, what: &str) -> (f32, usize) {
    let host = m.render_tile(tile, size);
    let dev = gpu.render_tile(m, tile, size);
    assert_eq!(host.data.len(), dev.data.len(), "{what}: length mismatch");

    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    let mut byte_diffs = 0usize;
    for (i, (h, d)) in host.data.iter().zip(&dev.data).enumerate() {
        let e = (h - d).abs();
        if e > worst {
            worst = e;
            worst_at = i;
        }
        if tilekiln::png::quantise(*h) != tilekiln::png::quantise(*d) {
            byte_diffs += 1;
        }
    }
    if worst > MAX_ABS_ERROR {
        let (x, y) = (worst_at as u32 % size, worst_at as u32 / size);
        let (h, d) = (host.data[worst_at], dev.data[worst_at]);
        panic!(
            "{what}: {worst:.3e} apart at ({x}, {y}), above the {MAX_ABS_ERROR:.1e} bound. \
             host {h:.9} vs device {d:.9}"
        );
    }
    // A byte difference means the value sat on a quantisation boundary,
    // which is rare but legitimate. A *lot* of them would mean the error
    // had grown to the scale of the output.
    let budget = (host.data.len() / 100).max(4);
    assert!(
        byte_diffs <= budget,
        "{what}: {byte_diffs} of {} pixels differ once quantised to 8 bits, \
         above the budget of {budget}. That is no longer rounding noise.",
        host.data.len()
    );
    (worst, byte_diffs)
}

#[test]
fn every_pattern_and_basis_stays_within_the_error_bound() {
    let gpu = gpu_or_skip!();
    eprintln!("device: {}", gpu.name);

    let mut checked = 0;
    let mut exact = 0;
    for basis in [Basis::Value, Basis::Gradient, Basis::Worley] {
        for pattern in [Pattern::Fractal, Pattern::Ridged, Pattern::Warped] {
            let m = Material {
                basis,
                pattern,
                frequency: 5.0,
                octaves: 4,
                sharpness: 3,
                warp: 0.7,
                contrast: 1.3,
                pivot: 0.5,
                seed: 12_345,
            };
            let (worst, bytes) = compare(
                &gpu,
                &m,
                TileId::new(0, 0),
                64,
                &format!("{basis:?}/{pattern:?}"),
            );
            let how_close = if worst == 0.0 {
                "exact".to_string()
            } else {
                format!(
                    "{worst:.2e}, {:.0}x below one 8-bit level",
                    QUANT_STEP / worst
                )
            };
            eprintln!("  {basis:?}/{pattern:?}: {how_close}, {bytes} bytes differ");
            if worst == 0.0 {
                exact += 1;
            }
            checked += 1;
        }
    }
    eprintln!("{exact} of {checked} combinations exact, all within {MAX_ABS_ERROR:.0e}");
}

#[test]
fn far_tiles_stay_within_the_error_bound() {
    // The coordinate split is what makes this work: the magnitude lives
    // in an integer cell, so the device never sees a large float.
    let gpu = gpu_or_skip!();
    let m = Material {
        basis: Basis::Gradient,
        pattern: Pattern::Ridged,
        frequency: 6.0,
        octaves: 5,
        sharpness: 3,
        contrast: 1.4,
        ..Default::default()
    };
    for shift in [0u32, 10, 20, 30, 40] {
        let t = 1i64 << shift;
        compare(&gpu, &m, TileId::new(t, -t), 48, &format!("tile 2^{shift}"));
    }
}

#[test]
fn awkward_parameters_stay_within_the_error_bound() {
    let gpu = gpu_or_skip!();
    for (i, m) in [
        // Non-power-of-two frequency, so the origin split has a
        // non-trivial fractional part.
        Material {
            frequency: 17.31,
            octaves: 6,
            ..Default::default()
        },
        // Single octave: no doubling at all.
        Material {
            frequency: 3.0,
            octaves: 1,
            ..Default::default()
        },
        // Frequency below one, so the whole tile sits inside one cell.
        Material {
            frequency: 0.25,
            octaves: 3,
            ..Default::default()
        },
        // Heavy contrast, so clamping is exercised on both sides.
        Material {
            frequency: 4.0,
            contrast: 12.0,
            pivot: 0.3,
            ..Default::default()
        },
        // Zero warp on the warped pattern.
        Material {
            pattern: Pattern::Warped,
            warp: 0.0,
            frequency: 4.0,
            ..Default::default()
        },
        // Sharpness of one: the loop body runs zero extra times.
        Material {
            pattern: Pattern::Ridged,
            sharpness: 1,
            frequency: 5.0,
            ..Default::default()
        },
    ]
    .into_iter()
    .enumerate()
    {
        compare(
            &gpu,
            &m,
            TileId::new(3, -7),
            64,
            &format!("awkward case {i}"),
        );
    }
}

#[test]
fn the_device_is_reproducible() {
    let gpu = gpu_or_skip!();
    let m = Material {
        frequency: 6.0,
        octaves: 5,
        ..Default::default()
    };
    let first = gpu.render(&m, 64);
    for i in 1..4 {
        let again = gpu.render(&m, 64);
        assert_eq!(first, again, "device run {i} differed from run 0");
    }
}

#[test]
fn odd_tile_sizes_stay_within_the_error_bound() {
    // The dispatch rounds up to whole 8x8 workgroups, so a size that is
    // not a multiple of eight exercises the bounds check.
    let gpu = gpu_or_skip!();
    let m = Material {
        frequency: 4.0,
        octaves: 3,
        ..Default::default()
    };
    for size in [1u32, 3, 7, 8, 9, 31, 33] {
        compare(&gpu, &m, TileId::new(0, 0), size, &format!("size {size}"));
    }
}

// =====================================================================
// The fixed-point path
// =====================================================================
//
// No bound here, and no tolerance. Every operation on both sides is an
// integer multiply, add or shift, so the results are equal or the
// implementation is wrong.

fn compare_exact(gpu: &Gpu, m: &Material, tile: TileId, size: u32, what: &str) {
    let host = m.render_tile_exact(tile, size);
    let dev = gpu.render_tile_exact(m, tile, size);
    assert_eq!(host.data.len(), dev.data.len(), "{what}: length mismatch");

    let mut differing = 0usize;
    let mut first = None;
    for (i, (h, d)) in host.data.iter().zip(&dev.data).enumerate() {
        if h.to_bits() != d.to_bits() {
            differing += 1;
            if first.is_none() {
                first = Some((i, *h, *d));
            }
        }
    }
    if let Some((i, h, d)) = first {
        let (x, y) = (i as u32 % size, i as u32 / size);
        panic!(
            "{what}: {differing} of {} pixels differ. First at ({x}, {y}): \
             host {h:.9} ({:08x}) vs device {d:.9} ({:08x})",
            host.data.len(),
            h.to_bits(),
            d.to_bits()
        );
    }
}

#[test]
fn the_fixed_point_path_is_bit_identical() {
    let gpu = gpu_or_skip!();
    let mut checked = 0;
    for basis in [Basis::Value, Basis::Gradient, Basis::Worley] {
        for pattern in [Pattern::Fractal, Pattern::Ridged, Pattern::Warped] {
            let m = Material {
                basis,
                pattern,
                frequency: 5.0,
                octaves: 4,
                sharpness: 3,
                warp: 0.7,
                contrast: 1.3,
                seed: 999,
                ..Default::default()
            };
            compare_exact(
                &gpu,
                &m,
                TileId::new(3, -7),
                64,
                &format!("{basis:?}/{pattern:?}"),
            );
            checked += 1;
        }
    }
    eprintln!("{checked} basis/pattern combinations bit-identical");
}

#[test]
fn the_fixed_point_path_is_bit_identical_on_far_tiles() {
    let gpu = gpu_or_skip!();
    let m = Material {
        basis: Basis::Gradient,
        pattern: Pattern::Ridged,
        frequency: 6.0,
        octaves: 5,
        sharpness: 3,
        ..Default::default()
    };
    // Out to the last index that does not alias, and one past it, since
    // a wrapped cell has to agree too.
    for &t in &[0i64, 1 << 10, 1 << 20, m.max_tile(), 1 << 40] {
        compare_exact(&gpu, &m, TileId::new(t, -t), 48, &format!("tile {t}"));
    }
}

#[test]
fn the_fixed_point_path_is_bit_identical_on_awkward_parameters() {
    let gpu = gpu_or_skip!();
    for (i, m) in [
        // Non-power-of-two frequency, so the origin split is untidy.
        Material {
            frequency: 17.31,
            octaves: 6,
            ..Default::default()
        },
        // Single octave: no doubling at all.
        Material {
            frequency: 3.0,
            octaves: 1,
            ..Default::default()
        },
        // Frequency below one, so the tile sits inside one cell.
        Material {
            frequency: 0.25,
            octaves: 3,
            ..Default::default()
        },
        // Heavy contrast, so the clamp is exercised at both ends.
        Material {
            frequency: 4.0,
            contrast: 12.0,
            pivot: 0.3,
            ..Default::default()
        },
        // Zero warp on the warped pattern.
        Material {
            pattern: Pattern::Warped,
            warp: 0.0,
            frequency: 4.0,
            ..Default::default()
        },
        // Sharpness of one: the inner loop runs zero extra times.
        Material {
            pattern: Pattern::Ridged,
            sharpness: 1,
            frequency: 5.0,
            ..Default::default()
        },
        // Worley at a high frequency, which is where the integer square
        // root does the most work and where a wrong rejection bound in
        // the restoring loop would show.
        Material {
            basis: Basis::Worley,
            frequency: 14.0,
            octaves: 5,
            ..Default::default()
        },
    ]
    .into_iter()
    .enumerate()
    {
        compare_exact(
            &gpu,
            &m,
            TileId::new(3, -7),
            64,
            &format!("awkward case {i}"),
        );
    }
}

#[test]
fn the_fixed_point_path_is_bit_identical_at_odd_tile_sizes() {
    let gpu = gpu_or_skip!();
    let m = Material {
        basis: Basis::Worley,
        frequency: 7.0,
        octaves: 3,
        ..Default::default()
    };
    // Sizes that do not fill the 8x8 workgroup, where the bounds check
    // in the shader decides which invocations write.
    for size in [1u32, 3, 7, 8, 9, 17, 64, 65] {
        compare_exact(&gpu, &m, TileId::new(1, 1), size, &format!("size {size}"));
    }
}
