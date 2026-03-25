use std::sync::Arc;

use glam::IVec3;
use wgpu::{
    CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, SurfaceConfiguration, TextureUsages, TextureViewDescriptor,
    util::DeviceExt,
};
use winit::window::Window;

use voxel_core::{
    camera::{Camera, CameraUniform},
    gen::{TerrainParams, generate_chunk},
    gpu::{GpuContext, GpuError, create_uniform_buffer, write_uniform},
};

use crate::mesh::build_chunk_mesh;
use crate::pipeline::{ChunkUniform, NaivePipeline};

/// All GPU resources for one renderable chunk.
#[allow(dead_code)]
struct ChunkDraw {
    vertex_buf: wgpu::Buffer,
    index_buf:  wgpu::Buffer,
    index_count: u32,
    chunk_buf:  wgpu::Buffer,
    chunk_bind_group: wgpu::BindGroup,
}

pub struct NaiveRenderer {
    // Window first — must outlive surface (drop order = declaration order, reversed)
    pub window: Arc<Window>,

    pub gpu: GpuContext,
    surface: wgpu::Surface<'static>,
    config: SurfaceConfiguration,

    pub camera_buf: wgpu::Buffer,
    #[allow(dead_code)]
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_bind_group: wgpu::BindGroup,

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    pipeline: NaivePipeline,
    chunk_draws: Vec<ChunkDraw>,
}

impl NaiveRenderer {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(window: Arc<Window>, width: u32, height: u32) -> Result<Self, GpuError> {
        let instance = GpuContext::create_instance();

        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("create surface");

        let gpu = GpuContext::from_surface(instance, &surface, wgpu::Features::empty())?;
        log::info!("GPU: {}", gpu.adapter_info());

        let caps = surface.get_capabilities(&gpu.adapter);
        let surface_format = caps.formats.iter()
            .find(|f| f.is_srgb()).copied()
            .unwrap_or(caps.formats[0]);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::AutoVsync
        };

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &config);

        // ── Camera uniform ────────────────────────────────────────────────
        let camera_buf = create_uniform_buffer::<CameraUniform>(&gpu.device, "camera uniform");

        let camera_bind_group_layout =
            gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(CameraUniform::SIZE),
                    },
                    count: None,
                }],
            });

        let camera_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bg"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        // ── Depth buffer ──────────────────────────────────────────────────
        let (depth_texture, depth_view) =
            Self::create_depth_texture(&gpu.device, width.max(1), height.max(1));

        // ── Pipeline ──────────────────────────────────────────────────────
        let pipeline = NaivePipeline::new(
            &gpu.device,
            surface_format,
            &camera_bind_group_layout,
            Self::DEPTH_FORMAT,
        );

        // ── Generate test world ───────────────────────────────────────────
        let chunk_draws = Self::generate_world(
            &gpu.device,
            &pipeline,
        );

        Ok(NaiveRenderer {
            window,
            gpu,
            surface,
            config,
            camera_buf,
            camera_bind_group_layout,
            camera_bind_group,
            depth_texture,
            depth_view,
            pipeline,
            chunk_draws,
        })
    }

    /// Generates a 5×4×5 chunk world, inserts all chunks into a World so
    /// inter-chunk face culling works, then meshes each chunk.
    fn generate_world(
        device: &wgpu::Device,
        pipeline: &NaivePipeline,
    ) -> Vec<ChunkDraw> {
        use voxel_core::world::World;

        let params = TerrainParams::default();

        // Range: x/z = -2..=2, y = -2..=1
        // sea_level=32, so surface is in chunk y=1 (world y=32..64).
        // y=-2 = deep stone, y=-1 = lower stone, y=0 = upper stone/dirt, y=1 = surface+grass
        let x_range = -2i32..=2;
        let y_range = -2i32..=1;
        let z_range = -2i32..=2;

        // Phase 1: generate and insert all chunks into the world
        let mut world = World::new();
        for cy in y_range.clone() {
            for cz in z_range.clone() {
                for cx in x_range.clone() {
                    let pos = IVec3::new(cx, cy, cz);
                    let chunk = generate_chunk(pos, &params);
                    world.insert_chunk(pos, chunk);
                }
            }
        }

        // Phase 2: mesh each chunk with cross-chunk neighbour awareness
        let mut draws = Vec::new();
        for cy in y_range {
            for cz in z_range.clone() {
                for cx in x_range.clone() {
                    let chunk_pos = IVec3::new(cx, cy, cz);
                    let chunk = world.get_chunk(&chunk_pos).unwrap();
                    let (verts, idx) = build_chunk_mesh(chunk, chunk_pos, &world);

                    if verts.is_empty() {
                        continue;
                    }

                    let vertex_buf = device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("chunk vbuf"),
                            contents: bytemuck::cast_slice(&verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    );

                    let index_buf = device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("chunk ibuf"),
                            contents: bytemuck::cast_slice(&idx),
                            usage: wgpu::BufferUsages::INDEX,
                        },
                    );

                    use voxel_core::world::CHUNK_SIZE_I;
                    let origin = [
                        (cx * CHUNK_SIZE_I) as f32,
                        (cy * CHUNK_SIZE_I) as f32,
                        (cz * CHUNK_SIZE_I) as f32,
                        0.0f32,
                    ];
                    let chunk_uniform = ChunkUniform { origin };

                    let chunk_buf = device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("chunk uniform"),
                            contents: bytemuck::bytes_of(&chunk_uniform),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        },
                    );

                    let chunk_bind_group =
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("chunk bg"),
                            layout: &pipeline.chunk_bind_group_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: chunk_buf.as_entire_binding(),
                            }],
                        });

                    draws.push(ChunkDraw {
                        vertex_buf,
                        index_buf,
                        index_count: idx.len() as u32,
                        chunk_buf,
                        chunk_bind_group,
                    });
                }
            }
        }

        log::info!("Generated {} chunk draw calls", draws.len());
        draws
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        let (tex, view) = Self::create_depth_texture(&self.gpu.device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
    }

    // ── Render ────────────────────────────────────────────────────────────────

    pub fn render(&mut self, camera: &Camera) -> Result<(), wgpu::SurfaceError> {
        // Upload camera uniform
        write_uniform(
            &self.gpu.queue,
            &self.camera_buf,
            &CameraUniform::from_camera(camera),
        );

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self.gpu.device.create_command_encoder(
            &CommandEncoderDescriptor { label: Some("naive frame") },
        );

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("naive pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color {
                            r: 0.53, g: 0.81, b: 0.98, a: 1.0, // sky blue
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);

            for draw in &self.chunk_draws {
                pass.set_bind_group(1, &draw.chunk_bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buf.slice(..));
                pass.set_index_buffer(draw.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..draw.index_count, 0, 0..1);
            }
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    #[allow(dead_code)]
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        (texture, view)
    }
}