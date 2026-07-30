//! Device synthesis via `wgpu`, matching the CPU path to 9e-6, and
//! exactly on drivers that do not fuse.
//!
//! Texture synthesis is embarrassingly parallel per pixel, so a device
//! is the obvious place to run it. The question worth asking is how
//! closely the answer matches the host, and this module is built to make
//! that gap as small as it can be made.
//!
//! # What the design buys
//!
//! WGSL specifies `+`, `-`, `*` and `sqrt` on `f32` exactly, and leaves
//! `sin`, `exp`, `log`, `pow` and `inverseSqrt` to the driver. The
//! synthesis path is therefore built from the former only: gradients
//! come from a hashed table rather than trigonometry, the interpolant is
//! a quintic polynomial, ridge sharpening is repeated multiplication
//! rather than `pow`, and lacunarity is fixed at 2 so octave scaling is
//! exact.
//!
//! The other half is the coordinate. WGSL has no `f64`, so carrying a
//! world coordinate as one float would force a choice between losing
//! large tile indices and being unmatchable against an `f64` host.
//! [`crate::noise::Lattice`] escapes that: the magnitude lives in an
//! integer cell and only the sub-cell offset is a float. Octave doubling
//! is then exact on both sides, because doubling a binary fraction
//! introduces no rounding and the carry is an integer add.
//!
//! # Bit-identity turns out to belong to the driver
//!
//! The goal was bit-identity. Whether it is reached depends on which
//! driver compiles the shader, which was not the answer this module was
//! built expecting.
//!
//! A shader compiler is free to fuse a multiply and an add into a single
//! `fma`, rounding once where the host rounds twice, and WGSL provides no
//! way to forbid it. Almost every line of the synthesis path is a
//! multiply followed by an add: the quintic interpolant, the gradient dot
//! product, the octave accumulation, the contrast remap.
//!
//! Two drivers, same shader, 64×64 tile, nine basis/pattern combinations:
//!
//! | driver | exact | worst error |
//! |---|---|---|
//! | llvmpipe (software Vulkan, LLVM 20.1.2) | 8 of 9 | 1.2e-7 |
//! | Apple M4 Max (Metal) | 0 of 9 | 7.2e-7 |
//!
//! llvmpipe reproduces the host bit for bit almost everywhere, so the
//! arithmetic really is expressed exactly and the discipline above really
//! does its job. Metal fuses, so it does not. Neither is wrong; WGSL
//! permits both.
//!
//! What can be promised portably is therefore a bound rather than
//! equality. Over 311,040 samples spanning every basis and pattern, five
//! parameter sets and tile indices past the lattice wrap point, the worst
//! disagreement on Metal was 8.9e-6. One level of the 8-bit output is
//! 3.9e-3, roughly 440 times larger, so the rendered image is the same
//! image: 5 of those samples sat close enough to a quantisation boundary
//! to land on a different byte.
//!
//! `tests/gpu.rs` pins that bound, reports which combinations came out
//! exact, and skips cleanly when no adapter is present. CI runs it on
//! llvmpipe, so both columns of that table stay honest.
//!
//! Guaranteed exactness on every driver would mean not using floats at
//! all. That is how `perturbation-kernel` underneath gets a bit-exact
//! device path: emulated integer multiplication and an integer reduction,
//! with no float rounding left for a compiler to reassociate. Fixed-point
//! synthesis is the route here too, and it is not built.

use bytemuck::{Pod, Zeroable};

use crate::exact::{Lat, Prepared};
use crate::material::{Material, Pattern, Tile, TileId};
use crate::noise::{Basis, Lattice};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    origin_cell_x: i32,
    origin_cell_y: i32,
    origin_frac_x: f32,
    origin_frac_y: f32,
    size: u32,
    seed: u32,
    octaves: u32,
    sharpness: u32,
    basis: u32,
    pattern: u32,
    step: f32,
    warp: f32,
    contrast: f32,
    pivot: f32,
    pad0: f32,
    pad1: f32,
}

/// Parameters for the fixed-point pipeline.
///
/// Every field is an integer, including the ones that were floats in
/// [`Params`]: the whole point of that path is that no float crosses to
/// the device except the output.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsExact {
    origin_cell_x: i32,
    origin_cell_y: i32,
    origin_frac_x: i32,
    origin_frac_y: i32,
    size: u32,
    seed: u32,
    octaves: u32,
    sharpness: u32,
    basis: u32,
    pattern: u32,
    step: i32,
    warp: i32,
    contrast: i32,
    pivot: i32,
    recip: i32,
    recip2: i32,
}

/// Why a device could not be used.
#[derive(Debug, Clone)]
pub struct Unavailable(pub String);

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no usable compute device: {}", self.0)
    }
}

impl std::error::Error for Unavailable {}

/// A device and the compiled synthesis pipeline.
///
/// Acquiring an adapter costs tens of milliseconds, so construct one and
/// keep it.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pipeline_exact: wgpu::ComputePipeline,
    /// Human-readable device description.
    pub name: String,
}

impl Gpu {
    /// Acquire a device and compile the pipeline.
    pub fn new() -> Result<Self, Unavailable> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| Unavailable(e.to_string()))?;

        let info = adapter.get_info();
        let name = format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type);
        let limits = adapter.limits();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tilekiln"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| Unavailable(e.to_string()))?;

        // A validation error inside a dispatch is otherwise delivered
        // asynchronously and lost.
        device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu validation: {e}")));

        let compile = |label: &str, src: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("render"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline = compile("synth", include_str!("shaders/synth.wgsl"));
        let pipeline_exact = compile("synth_exact", include_str!("shaders/synth_exact.wgsl"));

        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_exact,
            name,
        })
    }

    /// Render one tile on the device.
    ///
    /// Returns exactly what [`Material::render_tile`] returns on drivers
    /// that do not fuse a multiply and an add, and within 9e-6 of it on
    /// ones that do. Either way the gap is far below one level of the
    /// 8-bit output. See the module docs.
    pub fn render_tile(&self, m: &Material, tile: TileId, size: u32) -> Tile {
        // The host splits the tile origin, exactly as the CPU renderer
        // does, so the device never sees a large coordinate.
        let freq = m.frequency as f64;
        let ox = Lattice::split(tile.x as f64 * freq);
        let oy = Lattice::split(tile.y as f64 * freq);

        let params = Params {
            origin_cell_x: ox.cell,
            origin_cell_y: oy.cell,
            origin_frac_x: ox.frac,
            origin_frac_y: oy.frac,
            size,
            seed: m.seed,
            octaves: m.octaves,
            sharpness: m.sharpness,
            basis: match m.basis {
                Basis::Value => 0,
                Basis::Gradient => 1,
                Basis::Worley => 2,
            },
            pattern: match m.pattern {
                Pattern::Fractal => 0,
                Pattern::Ridged => 1,
                Pattern::Warped => 2,
            },
            step: m.frequency / size as f32,
            warp: m.warp,
            contrast: m.contrast,
            pivot: m.pivot,
            pad0: 0.0,
            pad1: 0.0,
        };

        self.dispatch(&self.pipeline, bytemuck::bytes_of(&params), size)
    }

    /// Render one tile with the fixed-point pipeline.
    ///
    /// Returns exactly what [`Material::render_tile_exact`] returns, on
    /// every driver. Nothing in that path is a float, so there is no
    /// rounding for a shader compiler to decide differently about.
    pub fn render_tile_exact(&self, m: &Material, tile: TileId, size: u32) -> Tile {
        let p = Prepared::new(m);
        let freq = m.frequency as f64;
        let ox = Lat::split(tile.x as f64 * freq);
        let oy = Lat::split(tile.y as f64 * freq);

        let params = ParamsExact {
            origin_cell_x: ox.cell,
            origin_cell_y: oy.cell,
            origin_frac_x: ox.frac,
            origin_frac_y: oy.frac,
            size,
            seed: p.seed,
            octaves: p.octaves,
            sharpness: p.sharpness,
            basis: match p.basis {
                Basis::Value => 0,
                Basis::Gradient => 1,
                Basis::Worley => 2,
            },
            pattern: match p.pattern {
                Pattern::Fractal => 0,
                Pattern::Ridged => 1,
                Pattern::Warped => 2,
            },
            step: crate::fixed::from_f32(m.frequency / size as f32),
            warp: p.warp,
            contrast: p.contrast,
            pivot: p.pivot,
            recip: p.recip,
            recip2: p.recip2,
        };

        self.dispatch(&self.pipeline_exact, bytemuck::bytes_of(&params), size)
    }

    /// Render the unit tile at the origin with the fixed-point pipeline.
    pub fn render_exact(&self, m: &Material, size: u32) -> Tile {
        self.render_tile_exact(m, TileId::new(0, 0), size)
    }

    /// Upload parameters, run one pipeline over a tile, read it back.
    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, params: &[u8], size: u32) -> Tile {
        let bytes = (size as u64) * (size as u64) * 4;
        let pbuf = wgpu::util::DeviceExt::create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: params,
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("synth"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pbuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("synth"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("synth"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = size.div_ceil(8);
            pass.dispatch_workgroups(groups, groups, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv().expect("readback channel").expect("buffer map");
        let view = slice.get_mapped_range().expect("mapped range");
        let data: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        staging.unmap();

        Tile { size, data }
    }

    /// Render the unit tile at the origin.
    pub fn render(&self, m: &Material, size: u32) -> Tile {
        self.render_tile(m, TileId::new(0, 0), size)
    }
}
