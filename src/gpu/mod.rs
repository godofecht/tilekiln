//! Device synthesis via `wgpu`, matching the CPU path to 9e-6.
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
//! # What it does not buy
//!
//! The original goal was bit-identity, and that goal is not met. A
//! shader compiler may fuse a multiply and an add into a single `fma`,
//! rounding once where the host rounds twice, and WGSL has no way to
//! forbid it. Almost every line here is a multiply followed by an add,
//! so the difference arises in several places at once; adding an
//! explicit `mul_add` on the host brings one of them into line and
//! leaves the others, which is how the cause was pinned down.
//!
//! Measured over 311,040 samples across every basis and pattern, five
//! parameter sets and tile indices past the lattice wrap point, the
//! worst disagreement was
//! 8.9e-6 in absolute terms. One level of the 8-bit output is 3.9e-3,
//! roughly 440 times larger, so the rendered image is the same image: 5
//! of those samples sat close enough to a quantisation boundary to land
//! on a different byte.
//!
//! `tests/gpu.rs` pins that bound and skips cleanly when no adapter is
//! present.
//!
//! Exactness is still reachable, but not this way. It needs integer
//! arithmetic, which is how `perturbation-kernel` gets a bit-exact
//! device path: emulated integer multiplication and an integer
//! reduction, with no float rounding to disagree about. Fixed-point
//! synthesis is the route here too, and it is not yet built.

use bytemuck::{Pod, Zeroable};

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

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("synth"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/synth.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("synth"),
            layout: None,
            module: &module,
            entry_point: Some("render"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            name,
        })
    }

    /// Render one tile on the device.
    ///
    /// Agrees with [`Material::render_tile`] to within 9e-6, which is
    /// far below one level of the 8-bit output. See the module docs for
    /// why it is not exact.
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

        let bytes = (size as u64) * (size as u64) * 4;
        let pbuf = wgpu::util::DeviceExt::create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
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
            layout: &self.pipeline.get_bind_group_layout(0),
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
            pass.set_pipeline(&self.pipeline);
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
