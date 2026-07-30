//! Write the images used in the README.
//!
//! Run with `cargo run --release --example gallery`. Everything lands
//! in `docs/img/`.
//!
//! The interesting one is `perturbed_*.png`. Sensitivity numbers are
//! easy to nod along to and hard to feel, so those strips render the
//! *same* nudge applied to a knob the analysis calls safe and a knob it
//! calls dangerous. If the measurement is worth anything, the
//! difference should be obvious by eye.

use std::fs;
use std::path::Path;

use tilekiln::analysis::{sensitivities, Settings};
use tilekiln::features::Feature;
use tilekiln::material::Pattern;
use tilekiln::png;
use tilekiln::{Basis, Material, Tile, TileId};

const OUT: &str = "docs/img";
// Sizes are kept modest on purpose. The PNG writer uses stored
// DEFLATE, so a file costs about its raw byte count; that is the price
// of having no image-codec dependency, and it is paid in the repository
// rather than at run time.
const SIZE: u32 = 224;

fn write(name: &str, bytes: &[u8]) {
    let path = Path::new(OUT).join(name);
    fs::write(&path, bytes).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    println!("  {:<28} {:>7} kB", name, bytes.len() / 1024);
}

/// Lay tiles out side by side into one greyscale image.
fn strip(tiles: &[Tile], gap: u32) -> (u32, u32, Vec<u8>) {
    let n = tiles.len() as u32;
    let s = tiles[0].size;
    let w = n * s + (n - 1) * gap;
    let mut px = vec![255u8; (w * s) as usize];
    for (i, t) in tiles.iter().enumerate() {
        let x0 = i as u32 * (s + gap);
        for y in 0..s {
            for x in 0..s {
                px[(y * w + x0 + x) as usize] = png::quantise(t.get(x, y));
            }
        }
    }
    (w, s, px)
}

fn presets() -> Vec<(&'static str, Material)> {
    vec![
        (
            "soft_clouds",
            Material {
                basis: Basis::Gradient,
                pattern: Pattern::Fractal,
                frequency: 2.0,
                octaves: 6,
                contrast: 1.1,
                ..Default::default()
            },
        ),
        (
            "cracked_rock",
            Material {
                basis: Basis::Gradient,
                pattern: Pattern::Ridged,
                frequency: 6.0,
                octaves: 5,
                sharpness: 3,
                contrast: 1.4,
                ..Default::default()
            },
        ),
        (
            "marbled_vein",
            Material {
                basis: Basis::Gradient,
                pattern: Pattern::Warped,
                frequency: 3.0,
                octaves: 5,
                warp: 0.9,
                contrast: 1.3,
                ..Default::default()
            },
        ),
        (
            "cellular",
            Material {
                basis: Basis::Worley,
                pattern: Pattern::Fractal,
                frequency: 5.0,
                octaves: 2,
                contrast: 1.2,
                ..Default::default()
            },
        ),
    ]
}

fn main() {
    fs::create_dir_all(OUT).expect("creating docs/img");
    println!("writing to {OUT}/\n");

    // ---- the presets -----------------------------------------------
    for (name, m) in presets() {
        write(&format!("{name}.png"), &m.render(SIZE).to_png());
    }

    // ---- seamlessness ----------------------------------------------
    // Four tiles in a 2x2 block. Adjacent tiles evaluate the same
    // continuous field at the same coordinates, so there is no seam to
    // hide and none should be visible at the joins.
    {
        let m = presets()[2].1;
        let s = SIZE;
        let mut px = vec![0u8; (s * 2 * s * 2) as usize];
        for (ty, tx) in [(0u32, 0u32), (0, 1), (1, 0), (1, 1)] {
            let t = m.render_tile(TileId::new(tx as i64, ty as i64), s);
            for y in 0..s {
                for x in 0..s {
                    let (gx, gy) = (tx * s + x, ty * s + y);
                    px[(gy * s * 2 + gx) as usize] = png::quantise(t.get(x, y));
                }
            }
        }
        write("tiling_2x2.png", &png::grey(s * 2, s * 2, &px));
    }

    // ---- far tiles keep their detail --------------------------------
    // The bug this guards against: with f32 coordinates, tile 2^20
    // rendered 2,207 distinct values out of 65,536.
    //
    // The indices stop at `max_tile()`. An earlier version of this strip
    // ran out to 2^40, which is past the point where the i32 lattice
    // cell wraps: the last tile came back byte-identical to the first
    // while looking perfectly detailed, and the caption underneath it
    // claimed they differed. Ask for the largest index that means
    // something instead.
    {
        let m = presets()[1].1;
        let limit = m.max_tile();
        let indices = [0i64, 1 << 10, 1 << 20, 1 << 26, limit];
        let tiles: Vec<Tile> = indices
            .iter()
            .map(|&t| m.render_tile(TileId::new(t, 0), 160))
            .collect();

        let distinct = tiles
            .iter()
            .map(|t| t.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        println!("\n  far tiles up to {limit}: {distinct}/5 distinct");
        assert_eq!(distinct, 5, "two of the far tiles are the same tile");

        let (w, h, px) = strip(&tiles, 8);
        write("far_tiles.png", &png::grey(w, h, &px));
    }

    // ---- what the analysis is actually measuring --------------------
    // Same 8% nudge, applied to the knob the analysis ranks safest and
    // the knob it ranks most dangerous, on the same material.
    {
        let base = presets()[1].1;
        let settings = Settings {
            samples: 2_000,
            size: 48,
            max_relative: 0.08,
            feature: Feature::EdgeDensity,
            ..Default::default()
        };
        let ranked = sensitivities(&base, &settings);
        let worst = ranked.first().expect("at least one knob");
        let best = ranked
            .iter()
            .rev()
            .find(|r| r.spread > 0.0)
            .unwrap_or(worst);

        let index_of = |name: &str| Material::KNOBS.iter().position(|k| *k == name).unwrap();
        println!(
            "\n  most sensitive knob: {:<10} spread {:.5}",
            worst.knob, worst.spread
        );
        println!(
            "  least sensitive:     {:<10} spread {:.5}\n",
            best.knob, best.spread
        );

        for (label, r) in [("stable", best), ("dangerous", worst)] {
            let k = index_of(r.knob);
            let v = base.knob(k).unwrap();

            // A percentage nudge is the wrong step for an integer knob:
            // 8% of `sharpness = 3` is 0.24, which rounds straight back
            // to 3, and the strip comes out as five identical tiles.
            // That is the same trap the analysis fell into before
            // quantised knobs got stochastic rounding, and it is just as
            // invisible here. Step whole units instead.
            let (values, step_note) = if Material::KNOB_QUANTISED[k] {
                let base_i = v.round() as i32;
                let vals: Vec<f32> = (-2..=2).map(|d| (base_i + d).max(1) as f32).collect();
                (vals, "+/-2 steps".to_string())
            } else {
                let vals: Vec<f32> = [-0.08f32, -0.04, 0.0, 0.04, 0.08]
                    .iter()
                    .map(|d| v * (1.0 + d))
                    .collect();
                (vals, "+/-8%".to_string())
            };

            let tiles: Vec<Tile> = values
                .iter()
                .map(|&nv| base.with_knob(k, nv).render(160))
                .collect();

            // A strip of identical tiles proves nothing, so check.
            let distinct = tiles
                .iter()
                .map(|t| t.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            println!(
                "  {label:<10} {:<10} {step_note:<12} {distinct}/5 tiles distinct",
                r.knob
            );
            assert!(
                distinct >= 4,
                "the {label} strip has only {distinct} distinct tiles, so it \
                 shows nothing"
            );

            let (w, h, px) = strip(&tiles, 8);
            write(&format!("perturbed_{label}.png"), &png::grey(w, h, &px));
        }
    }

    println!("\ndone");
}
