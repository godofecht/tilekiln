//! Minimal PNG writer, no dependencies.
//!
//! A texture crate that cannot produce a picture is not much use, and
//! pulling an image codec in for one output format would be a poor
//! trade for a library whose whole argument is that it has no hidden
//! state.
//!
//! DEFLATE is used in *stored* mode: every block is uncompressed. That
//! is a legal zlib stream, needs no compressor, and costs about 1.001x
//! the raw byte count. For a 512x512 tile that is 260 kB instead of
//! perhaps 90 kB, which is the right trade for deleting a dependency
//! from a crate whose output is normally consumed in memory rather than
//! written to disk.

const CRC_POLY: u32 = 0xEDB8_8320;

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (CRC_POLY & mask);
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in bytes {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// Wrap raw bytes in a zlib stream of stored DEFLATE blocks.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 16);
    // CMF/FLG for deflate, 32 KiB window, no preset dictionary.
    out.extend_from_slice(&[0x78, 0x01]);
    let mut rest = raw;
    loop {
        let take = rest.len().min(65535);
        let last = take == rest.len();
        out.push(if last { 1 } else { 0 });
        out.extend_from_slice(&(take as u16).to_le_bytes());
        out.extend_from_slice(&(!(take as u16)).to_le_bytes());
        out.extend_from_slice(&rest[..take]);
        if last {
            break;
        }
        rest = &rest[take..];
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

fn encode(width: u32, height: u32, colour_type: u8, channels: usize, pixels: &[u8]) -> Vec<u8> {
    assert_eq!(
        pixels.len(),
        (width as usize) * (height as usize) * channels,
        "pixel buffer does not match the declared dimensions"
    );

    // Each scanline is prefixed with a filter byte; 0 means "none".
    let stride = width as usize * channels;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for y in 0..height as usize {
        raw.push(0);
        raw.extend_from_slice(&pixels[y * stride..(y + 1) * stride]);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, colour_type, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}

/// Encode an 8-bit greyscale image.
pub fn grey(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    encode(width, height, 0, 1, pixels)
}

/// Encode an 8-bit RGB image.
pub fn rgb(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    encode(width, height, 2, 3, pixels)
}

/// Encode a greyscale buffer as RGB, for viewers that handle colour
/// PNGs more consistently than greyscale ones.
pub fn rgb_from_grey(width: u32, height: u32, grey_px: &[u8]) -> Vec<u8> {
    let mut px = Vec::with_capacity(grey_px.len() * 3);
    for &g in grey_px {
        px.extend_from_slice(&[g, g, g]);
    }
    rgb(width, height, &px)
}

/// Quantise a `[0, 1]` sample to a byte, clamping out-of-range input.
#[inline]
pub fn quantise(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_known_value() {
        // The PNG specification's own worked example.
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }

    #[test]
    fn adler_matches_the_known_value() {
        // RFC 1950's example.
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn output_has_the_right_shape() {
        let px = vec![128u8; 4 * 4];
        let png = grey(4, 4, &px);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR, IDAT and IEND must all be present, in that order.
        let find = |needle: &[u8]| png.windows(4).position(|w| w == needle);
        let (a, b, c) = (find(b"IHDR"), find(b"IDAT"), find(b"IEND"));
        assert!(a < b && b < c, "chunks out of order: {a:?} {b:?} {c:?}");
        assert_eq!(&png[png.len() - 4..], &0xAE42_6082u32.to_be_bytes());
    }

    #[test]
    fn multi_block_images_stay_valid() {
        // Larger than one 65535-byte stored block, so the block
        // chaining is exercised.
        let px = vec![7u8; 300 * 300];
        let png = grey(300, 300, &px);
        assert!(png.len() > 65_535);
        assert_eq!(&png[png.len() - 4..], &0xAE42_6082u32.to_be_bytes());
    }

    #[test]
    fn quantise_pins_its_endpoints() {
        assert_eq!(quantise(0.0), 0);
        assert_eq!(quantise(1.0), 255);
        assert_eq!(quantise(-5.0), 0);
        assert_eq!(quantise(5.0), 255);
    }
}
