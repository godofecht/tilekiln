// Device mirror of the fixed-point synthesis path in `src/exact.rs`.
//
// The floating-point shader next door agrees with the host on llvmpipe
// and not on Metal, because a driver may fuse a multiply and an add into
// one `fma` and WGSL cannot forbid it. This one has no floats to fuse.
// Every value below is a Q4.27 integer: the number divided by 2^27.
//
// Two things need spelling out that Rust gets for free.
//
//   1. A Q4.27 multiply needs the full 64-bit product before shifting it
//      back down, and WGSL has no integer wider than 32 bits. `umul32`
//      builds it from four 16-bit partial products.
//   2. Worley distance needs a square root of a value up to 2^57. There
//      is no exact integer root in WGSL either, so `isqrt_u64` runs the
//      restoring binary algorithm over a `U64` pair.
//
// The float shader's rule about avoiding `sin`, `exp`, `pow` and
// `inverseSqrt` does not apply here, because nothing in this file is a
// float until the last line.

const SHIFT: u32 = 27u;
const ONE: i32 = 134217728;      // 1 << 27
const FRAC_MASK: u32 = 134217727u;

struct Params {
    origin_cell_x: i32,
    origin_cell_y: i32,
    origin_frac_x: i32,
    origin_frac_y: i32,

    size: u32,
    seed: u32,
    octaves: u32,
    sharpness: u32,

    basis: u32,      // 0 value, 1 gradient, 2 worley
    pattern: u32,    // 0 fractal, 1 ridged, 2 warped
    step: i32,       // frequency / size, in Q4.27
    warp: i32,

    contrast: i32,
    pivot: i32,
    recip: i32,      // 1 / fractal_norm(octaves)
    recip2: i32,     // 1 / fractal_norm(2), for the warp fields
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

// ---------------------------------------------------------------------
// 64-bit arithmetic, built from u32 pairs
// ---------------------------------------------------------------------

struct U64 {
    hi: u32,
    lo: u32,
}

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(a.hi + b.hi + carry, lo);
}

fn u64_sub(a: U64, b: U64) -> U64 {
    let lo = a.lo - b.lo;
    let borrow = select(0u, 1u, a.lo < b.lo);
    return U64(a.hi - b.hi - borrow, lo);
}

fn u64_ge(a: U64, b: U64) -> bool {
    if (a.hi != b.hi) {
        return a.hi > b.hi;
    }
    return a.lo >= b.lo;
}

fn u64_is_zero(a: U64) -> bool {
    return a.hi == 0u && a.lo == 0u;
}

fn u64_gt(a: U64, b: U64) -> bool {
    return !u64_ge(b, a);
}

// Shift right by 1 or 2. Only those two are needed, and writing them
// separately avoids a variable shift distance spanning the word
// boundary.
fn u64_shr1(a: U64) -> U64 {
    return U64(a.hi >> 1u, (a.lo >> 1u) | (a.hi << 31u));
}

fn u64_shr2(a: U64) -> U64 {
    return U64(a.hi >> 2u, (a.lo >> 2u) | (a.hi << 30u));
}

// Full 64-bit product of two u32, from four 16-bit partial products.
fn umul32(a: u32, b: u32) -> U64 {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;

    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;

    // The two middle products can carry out of 32 bits when summed.
    let mid = p01 + p10;
    let mid_carry = select(0u, 0x10000u, mid < p01);

    let lo = p00 + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < p00);

    let hi = p11 + (mid >> 16u) + mid_carry + lo_carry;
    return U64(hi, lo);
}

// Floor of the square root of a 64-bit value. Restoring binary square
// root: one result bit per iteration, highest first.
fn isqrt_u64(n: U64) -> u32 {
    var num = n;
    var res = U64(0u, 0u);
    var bit = U64(0x40000000u, 0u);   // 1 << 62

    // Largest even power of two not above `num`. Terminates for
    // num = 0 as well, since 1 << 62 shifts down to zero in 31 steps.
    loop {
        if (!u64_gt(bit, num)) {
            break;
        }
        bit = u64_shr2(bit);
    }

    loop {
        if (u64_is_zero(bit)) {
            break;
        }
        let t = u64_add(res, bit);
        if (u64_ge(num, t)) {
            num = u64_sub(num, t);
            res = u64_add(u64_shr1(res), bit);
        } else {
            res = u64_shr1(res);
        }
        bit = u64_shr2(bit);
    }
    return res.lo;
}

// ---------------------------------------------------------------------
// fixed.rs
// ---------------------------------------------------------------------

// Absolute value as a magnitude, correct at i32 minimum.
fn iabs(a: i32) -> u32 {
    let s = a >> 31u;
    return bitcast<u32>((a ^ s) - s);
}

// Q4.27 multiply, truncating towards zero.
fn fx_mul(a: i32, b: i32) -> i32 {
    let p = umul32(iabs(a), iabs(b));
    // >> 27 of a 64-bit value: 32 - 27 = 5.
    let r = bitcast<i32>((p.hi << 5u) | (p.lo >> SHIFT));
    let neg = ((a ^ b) >> 31u) != 0;
    return select(r, -r, neg);
}

// Q4.27 square root of a non-negative value.
fn fx_sqrt(a: i32) -> i32 {
    let u = bitcast<u32>(a);
    // a << 27, as a 64-bit pair.
    let n = U64(u >> 5u, u << SHIFT);
    return bitcast<i32>(isqrt_u64(n));
}

// ---------------------------------------------------------------------
// hash.rs, unchanged: it was always integer
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

// A hashed value in [-1, 1). Shifts only: the 23 bits an f32 mantissa
// would have carried land straight in the top of the fraction.
fn signed_fx(h: u32) -> i32 {
    return bitcast<i32>((h >> 9u) << 5u) - ONE;
}

// ---------------------------------------------------------------------
// exact.rs: Lat
// ---------------------------------------------------------------------

struct Lat {
    cell: i32,
    frac: i32,
}

fn lat_renormalise(l: Lat) -> Lat {
    let carry = l.frac >> SHIFT;
    return Lat(l.cell + carry, l.frac - (carry << SHIFT));
}

fn lat_offset(l: Lat, delta: i32) -> Lat {
    return lat_renormalise(Lat(l.cell, l.frac + delta));
}

// Advance by `n` steps of `step`. The product exceeds 32 bits on a wide
// tile at a high frequency, so it is taken at full width and split.
fn lat_step(l: Lat, n: u32, step: i32) -> Lat {
    let p = umul32(n, bitcast<u32>(step));
    let cell_delta = bitcast<i32>((p.hi << 5u) | (p.lo >> SHIFT));
    let frac_delta = bitcast<i32>(p.lo & FRAC_MASK);
    return lat_renormalise(Lat(l.cell + cell_delta, l.frac + frac_delta));
}

fn lat_double(l: Lat) -> Lat {
    let d = l.frac << 1u;
    let carry = d >> SHIFT;
    return Lat(l.cell * 2 + carry, d - (carry << SHIFT));
}

// ---------------------------------------------------------------------
// exact.rs: primitives
// ---------------------------------------------------------------------

fn smootherstep(t: i32) -> i32 {
    let t2 = fx_mul(t, t);
    let t3 = fx_mul(t2, t);
    let inner = fx_mul(t, 6 * ONE) - 15 * ONE;
    let outer = fx_mul(t, inner) + 10 * ONE;
    return fx_mul(t3, outer);
}

fn lerp_fx(a: i32, b: i32, t: i32) -> i32 {
    return a + fx_mul(t, b - a);
}

fn gradient_x(h: u32) -> i32 {
    let i = h & 7u;
    if (i == 0u) { return ONE; }
    if (i == 1u) { return -ONE; }
    if (i == 2u) { return 0; }
    if (i == 3u) { return 0; }
    if (i == 4u) { return ONE; }
    if (i == 5u) { return -ONE; }
    if (i == 6u) { return ONE; }
    return -ONE;
}

fn gradient_y(h: u32) -> i32 {
    let i = h & 7u;
    if (i == 0u) { return 0; }
    if (i == 1u) { return 0; }
    if (i == 2u) { return ONE; }
    if (i == 3u) { return -ONE; }
    if (i == 4u) { return ONE; }
    if (i == 5u) { return ONE; }
    if (i == 6u) { return -ONE; }
    return -ONE;
}

fn value2(lx: Lat, ly: Lat, seed: u32) -> i32 {
    let ix = lx.cell;
    let iy = ly.cell;
    let u = smootherstep(lx.frac);
    let v = smootherstep(ly.frac);

    let c00 = signed_fx(hash2(ix, iy, seed));
    let c10 = signed_fx(hash2(ix + 1, iy, seed));
    let c01 = signed_fx(hash2(ix, iy + 1, seed));
    let c11 = signed_fx(hash2(ix + 1, iy + 1, seed));

    return lerp_fx(lerp_fx(c00, c10, u), lerp_fx(c01, c11, u), v);
}

fn gradient2(lx: Lat, ly: Lat, seed: u32) -> i32 {
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

    let n00 = fx_mul(gradient_x(h00), xf) + fx_mul(gradient_y(h00), yf);
    let n10 = fx_mul(gradient_x(h10), xf - ONE) + fx_mul(gradient_y(h10), yf);
    let n01 = fx_mul(gradient_x(h01), xf) + fx_mul(gradient_y(h01), yf - ONE);
    let n11 = fx_mul(gradient_x(h11), xf - ONE) + fx_mul(gradient_y(h11), yf - ONE);

    return lerp_fx(lerp_fx(n00, n10, u), lerp_fx(n01, n11, u), v);
}

fn worley2(lx: Lat, ly: Lat, seed: u32) -> i32 {
    let xi = lx.cell;
    let yi = ly.cell;
    var best = 15 * ONE;

    for (var oy = -1; oy <= 1; oy = oy + 1) {
        for (var ox = -1; ox <= 1; ox = ox + 1) {
            let cx = xi + ox;
            let cy = yi + oy;
            let jx = (signed_fx(hash2(cx, cy, seed)) >> 1u) + ONE / 2;
            let jy = (signed_fx(hash3(cx, cy, 1, seed)) >> 1u) + ONE / 2;
            let dx = ox * ONE + jx - lx.frac;
            let dy = oy * ONE + jy - ly.frac;
            let d2 = fx_mul(dx, dx) + fx_mul(dy, dy);
            if (d2 < best) {
                best = d2;
            }
        }
    }
    return fx_sqrt(best);
}

fn basis_at(b: u32, lx: Lat, ly: Lat, seed: u32) -> i32 {
    if (b == 0u) {
        return value2(lx, ly, seed);
    }
    if (b == 1u) {
        return gradient2(lx, ly, seed);
    }
    return worley2(lx, ly, seed) * 2 - ONE;
}

// ---------------------------------------------------------------------
// exact.rs: fractal constructions
// ---------------------------------------------------------------------

fn fbm(b: u32, lx: Lat, ly: Lat, octaves: u32, seed: u32, recip: i32) -> i32 {
    var sum = 0;
    var amp = ONE;
    var cx = lx;
    var cy = ly;

    let n = max(octaves, 1u);
    for (var o = 0u; o < n; o = o + 1u) {
        sum = sum + fx_mul(amp, basis_at(b, cx, cy, seed + o * 0x9e3779b1u));
        amp = amp >> 1u;
        cx = lat_double(cx);
        cy = lat_double(cy);
    }
    return fx_mul(sum, recip);
}

fn ridged(b: u32, lx: Lat, ly: Lat, octaves: u32, sharpness: u32, seed: u32, recip: i32) -> i32 {
    var sum = 0;
    var amp = ONE;
    var cx = lx;
    var cy = ly;

    let n = max(octaves, 1u);
    let sh = max(sharpness, 1u);
    for (var o = 0u; o < n; o = o + 1u) {
        let v = basis_at(b, cx, cy, seed + o * 0x9e3779b1u);
        let r = ONE - abs(v);
        var sharp = r;
        for (var k = 1u; k < sh; k = k + 1u) {
            sharp = fx_mul(sharp, r);
        }
        sum = sum + fx_mul(amp, sharp);
        amp = amp >> 1u;
        cx = lat_double(cx);
        cy = lat_double(cy);
    }
    return fx_mul(sum, recip);
}

// 5.2, 1.3, 1.7 and 9.2 in Q4.27, the same constants `exact.rs` derives
// from the f32 literals.
const WARP_X0: i32 = 697932160;   // 5.2
const WARP_Y0: i32 = 174483040;   // 1.3
const WARP_X1: i32 = 228170144;   // 1.7
const WARP_Y1: i32 = 1234803072;  // 9.2

fn warped(b: u32, lx: Lat, ly: Lat, octaves: u32, strength: i32, seed: u32,
          recip: i32, recip2: i32) -> i32 {
    let wx = fbm(0u, lat_offset(lx, WARP_X0), lat_offset(ly, WARP_Y0),
                 2u, seed ^ 0x5f356495u, recip2);
    let wy = fbm(0u, lat_offset(lx, WARP_X1), lat_offset(ly, WARP_Y1),
                 2u, seed ^ 0x3c6ef372u, recip2);
    return fbm(b,
               lat_offset(lx, fx_mul(strength, wx)),
               lat_offset(ly, fx_mul(strength, wy)),
               octaves, seed, recip);
}

// ---------------------------------------------------------------------
// exact.rs: sample and render_tile_exact
// ---------------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.size;
    if (gid.x >= size || gid.y >= size) {
        return;
    }

    let ox = Lat(params.origin_cell_x, params.origin_frac_x);
    let oy = Lat(params.origin_cell_y, params.origin_frac_y);
    let lx = lat_step(ox, gid.x, params.step);
    let ly = lat_step(oy, gid.y, params.step);

    var raw: i32;
    if (params.pattern == 0u) {
        raw = fbm(params.basis, lx, ly, params.octaves, params.seed, params.recip);
    } else if (params.pattern == 1u) {
        raw = ridged(params.basis, lx, ly, params.octaves, params.sharpness,
                     params.seed, params.recip);
    } else {
        raw = warped(params.basis, lx, ly, params.octaves, params.warp,
                     params.seed, params.recip, params.recip2);
    }

    var unit: i32;
    if (params.pattern == 1u) {
        unit = raw;
    } else {
        unit = (raw >> 1u) + ONE / 2;
    }

    let v = clamp(fx_mul(unit - params.pivot, params.contrast) + params.pivot, 0, ONE);
    // The only float in the file. i32 to f32 is IEEE round-to-nearest and
    // the divisor is a power of two, so both are exactly specified.
    out[gid.y * size + gid.x] = f32(v) / f32(ONE);
}
