// Device mirror of the CPU synthesis path.
//
// Every function here corresponds line for line to something in
// `src/hash.rs`, `src/noise.rs` or `src/material.rs`, and the
// correspondence is the point: `tests/gpu.rs` holds the two paths to
// 9e-6 of each other, some 440x below one level of the 8-bit output.
//
// On llvmpipe it is exact for 8 of 9 basis/pattern combinations, so the
// arithmetic below really is expressed exactly. Apple's Metal compiler
// fuses a multiply and an add into an `fma` and nothing here can stop it,
// which is where the remaining error comes from. See `src/gpu/mod.rs`.
//
// Three rules keep the error at that scale rather than a visible one,
// and breaking any of them widens it silently.
//
//   1. Only `+`, `-`, `*` and `sqrt` on f32. WGSL specifies those
//      exactly; `sin`, `exp`, `log`, `pow` and `inverseSqrt` are left to
//      the driver, so one of them anywhere and the low bits diverge.
//   2. Coordinates travel as (integer cell, f32 fraction). WGSL has no
//      f64, so a single float coordinate would either lose large tile
//      indices or be unmatchable against an f64 host.
//   3. No dynamically indexed writes to function-local arrays. FXC, the
//      compiler the DX12 backend uses, refuses them: registers are not
//      addressable.

struct Params {
    // Tile origin, pre-split on the host.
    origin_cell_x: i32,
    origin_cell_y: i32,
    origin_frac_x: f32,
    origin_frac_y: f32,

    size: u32,
    seed: u32,
    octaves: u32,
    sharpness: u32,

    basis: u32,      // 0 value, 1 gradient, 2 worley
    pattern: u32,    // 0 fractal, 1 ridged, 2 warped
    step: f32,       // frequency / size
    warp: f32,

    contrast: f32,
    pivot: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

// ---------------------------------------------------------------------
// hash.rs
// ---------------------------------------------------------------------

fn mix_u32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash2(x: i32, y: i32, seed: u32) -> u32 {
    let h = (bitcast<u32>(x) * 0x27d4eb2du)
          ^ (bitcast<u32>(y) * 0x165667b1u)
          ^ (seed * 0x9e3779b1u);
    return mix_u32(h);
}

fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let h = (bitcast<u32>(x) * 0x27d4eb2du)
          ^ (bitcast<u32>(y) * 0x165667b1u)
          ^ (bitcast<u32>(z) * 0x0d2f1b3du)
          ^ (seed * 0x9e3779b1u);
    return mix_u32(h);
}

// Exact: 23 mantissa bits under a zero exponent, then subtract one. No
// division, so no rounding, so nothing for the host and device to
// disagree about.
fn unit_f32(h: u32) -> f32 {
    return bitcast<f32>((h >> 9u) | 0x3f800000u) - 1.0;
}

fn signed_f32(h: u32) -> f32 {
    return unit_f32(h) * 2.0 - 1.0;
}

// ---------------------------------------------------------------------
// noise.rs: Lattice
// ---------------------------------------------------------------------

struct Lat {
    cell: i32,
    frac: f32,
}

fn lat_renormalise(l: Lat) -> Lat {
    let fl = floor(l.frac);
    return Lat(l.cell + i32(fl), l.frac - fl);
}

fn lat_offset(l: Lat, delta: f32) -> Lat {
    return lat_renormalise(Lat(l.cell, l.frac + delta));
}

// Doubling a binary fraction is exact and the carry is an integer add,
// which is what lets the octave loop stay in lockstep with the host.
fn lat_double(l: Lat) -> Lat {
    let d = l.frac * 2.0;
    let fl = floor(d);
    return Lat(l.cell * 2 + i32(fl), d - fl);
}

// ---------------------------------------------------------------------
// noise.rs: primitives
// ---------------------------------------------------------------------

fn smootherstep(t: f32) -> f32 {
    // Horner form, matching the host exactly. The expanded polynomial
    // would round differently.
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp1(a: f32, b: f32, t: f32) -> f32 {
    // `a + t*(b - a)`: one fewer rounding than the symmetric form, and
    // exact at both endpoints.
    return a + t * (b - a);
}

// Eight gradients from a hash. A table rather than trigonometry, which
// keeps `sin` out of the pipeline. The diagonals stay unnormalised at
// length sqrt(2), exactly as Perlin's improved noise leaves them.
fn gradient_x(h: u32) -> f32 {
    let i = h & 7u;
    if (i == 0u) { return 1.0; }
    if (i == 1u) { return -1.0; }
    if (i == 2u) { return 0.0; }
    if (i == 3u) { return 0.0; }
    if (i == 4u) { return 1.0; }
    if (i == 5u) { return -1.0; }
    if (i == 6u) { return 1.0; }
    return -1.0;
}

fn gradient_y(h: u32) -> f32 {
    let i = h & 7u;
    if (i == 0u) { return 0.0; }
    if (i == 1u) { return 0.0; }
    if (i == 2u) { return 1.0; }
    if (i == 3u) { return -1.0; }
    if (i == 4u) { return 1.0; }
    if (i == 5u) { return 1.0; }
    if (i == 6u) { return -1.0; }
    return -1.0;
}

fn value2_at(lx: Lat, ly: Lat, seed: u32) -> f32 {
    let ix = lx.cell;
    let iy = ly.cell;
    let u = smootherstep(lx.frac);
    let v = smootherstep(ly.frac);

    let c00 = signed_f32(hash2(ix, iy, seed));
    let c10 = signed_f32(hash2(ix + 1, iy, seed));
    let c01 = signed_f32(hash2(ix, iy + 1, seed));
    let c11 = signed_f32(hash2(ix + 1, iy + 1, seed));

    return lerp1(lerp1(c00, c10, u), lerp1(c01, c11, u), v);
}

fn gradient2_at(lx: Lat, ly: Lat, seed: u32) -> f32 {
    let ix = lx.cell;
    let iy = ly.cell;
    let xf = lx.frac;
    let yf = ly.frac;
    let u = smootherstep(xf);
    let v = smootherstep(yf);

    let h00 = hash2(ix, iy, seed);
    let h10 = hash2(ix + 1, iy, seed);
    let h01 = hash2(ix, iy + 1, seed);
    let h11 = hash2(ix + 1, iy + 1, seed);

    let n00 = gradient_x(h00) * xf + gradient_y(h00) * yf;
    let n10 = gradient_x(h10) * (xf - 1.0) + gradient_y(h10) * yf;
    let n01 = gradient_x(h01) * xf + gradient_y(h01) * (yf - 1.0);
    let n11 = gradient_x(h11) * (xf - 1.0) + gradient_y(h11) * (yf - 1.0);

    return lerp1(lerp1(n00, n10, u), lerp1(n01, n11, u), v);
}

fn worley2_at(lx: Lat, ly: Lat, seed: u32) -> f32 {
    let xi = lx.cell;
    let yi = ly.cell;
    var best = 3.4028235e38;

    for (var oy = -1; oy <= 1; oy = oy + 1) {
        for (var ox = -1; ox <= 1; ox = ox + 1) {
            let cx = xi + ox;
            let cy = yi + oy;
            let h = hash2(cx, cy, seed);
            // Relative to the sample cell, so nothing large enters an
            // f32.
            let dx = f32(ox) + (signed_f32(h) * 0.5 + 0.5) - lx.frac;
            let dy = f32(oy) + (signed_f32(hash3(cx, cy, 1, seed)) * 0.5 + 0.5) - ly.frac;
            let d2 = dx * dx + dy * dy;
            if (d2 < best) {
                best = d2;
            }
        }
    }
    // sqrt is correctly rounded in IEEE-754 and WGSL inherits that.
    // inverseSqrt is not, and must not be substituted here.
    return sqrt(best);
}

fn basis_at(basis: u32, lx: Lat, ly: Lat, seed: u32) -> f32 {
    if (basis == 0u) {
        return value2_at(lx, ly, seed);
    }
    if (basis == 1u) {
        return gradient2_at(lx, ly, seed);
    }
    return worley2_at(lx, ly, seed) * 2.0 - 1.0;
}

// ---------------------------------------------------------------------
// noise.rs: fractal constructions
// ---------------------------------------------------------------------

fn fbm_at(basis: u32, lx: Lat, ly: Lat, octaves: u32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var norm = 0.0;
    var cx = lx;
    var cy = ly;

    let n = max(octaves, 1u);
    for (var o = 0u; o < n; o = o + 1u) {
        sum = sum + amp * basis_at(basis, cx, cy, seed + o * 0x9e3779b1u);
        norm = norm + amp;
        amp = amp * 0.5;
        cx = lat_double(cx);
        cy = lat_double(cy);
    }
    return sum / norm;
}

fn ridged_at(basis: u32, lx: Lat, ly: Lat, octaves: u32, sharpness: u32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var norm = 0.0;
    var cx = lx;
    var cy = ly;

    let n = max(octaves, 1u);
    let sh = max(sharpness, 1u);
    for (var o = 0u; o < n; o = o + 1u) {
        let v = basis_at(basis, cx, cy, seed + o * 0x9e3779b1u);
        let r = 1.0 - abs(v);
        // Integer exponent by repeated multiplication: `pow` is not
        // exactly specified in WGSL.
        var sharp = r;
        for (var k = 1u; k < sh; k = k + 1u) {
            sharp = sharp * r;
        }
        sum = sum + amp * sharp;
        norm = norm + amp;
        amp = amp * 0.5;
        cx = lat_double(cx);
        cy = lat_double(cy);
    }
    return sum / norm;
}

fn warped_at(basis: u32, lx: Lat, ly: Lat, octaves: u32, strength: f32, seed: u32) -> f32 {
    let wx = fbm_at(0u, lat_offset(lx, 5.2), lat_offset(ly, 1.3), 2u, seed ^ 0x5f356495u);
    let wy = fbm_at(0u, lat_offset(lx, 1.7), lat_offset(ly, 9.2), 2u, seed ^ 0x3c6ef372u);
    return fbm_at(
        basis,
        lat_offset(lx, strength * wx),
        lat_offset(ly, strength * wy),
        octaves,
        seed,
    );
}

// ---------------------------------------------------------------------
// material.rs: Material::sample_lattice and render_tile
// ---------------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size || gid.y >= size) {
        return;
    }

    let ox = Lat(params.origin_cell_x, params.origin_frac_x);
    let oy = Lat(params.origin_cell_y, params.origin_frac_y);
    let lx = lat_offset(ox, f32(gid.x) * params.step);
    let ly = lat_offset(oy, f32(gid.y) * params.step);

    var raw: f32;
    if (params.pattern == 0u) {
        raw = fbm_at(params.basis, lx, ly, params.octaves, params.seed);
    } else if (params.pattern == 1u) {
        raw = ridged_at(params.basis, lx, ly, params.octaves, params.sharpness, params.seed);
    } else {
        raw = warped_at(params.basis, lx, ly, params.octaves, params.warp, params.seed);
    }

    // Only the signed patterns are remapped; `ridged_at` already returns
    // [0, 1] and remapping it again would compress it into [0.5, 1.0].
    var unit: f32;
    if (params.pattern == 1u) {
        unit = raw;
    } else {
        unit = raw * 0.5 + 0.5;
    }

    let v = (unit - params.pivot) * params.contrast + params.pivot;
    out[gid.y * size + gid.x] = clamp(v, 0.0, 1.0);
}
