/// OptimisedRenderer — GPU surface + vertex-pulling render pass.
/// Uses the vertex pool's storage buffer and per-chunk draw calls.
/// Build order 4 will replace the draw loop with multi_draw_indirect.
use std::sync::Arc;

use wgpu::{
    CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, SurfaceConfiguration, TextureUsages, TextureViewDescriptor,
};
use winit::window::Window;

use voxel_core::{
    camera::{Camera, CameraUniform},
    gpu::{GpuContext, GpuError, create_uniform_buffer, write_uniform},
    input::RayHit,
};

use crate::pipeline::OptimisedPipeline;
use crate::world_manager::{WorldExtent, WorldManager};

pub struct OptimisedRenderer {
    pub window: Arc<Window>,

    pub gpu:     GpuContext,
    surface:     wgpu::Surface<'static>,
    config:      SurfaceConfiguration,

    pub camera_buf:          wgpu::Buffer,
    pub camera_bind_group:   wgpu::BindGroup,
    #[allow(dead_code)]
    camera_bgl:              wgpu::BindGroupLayout,

    depth_texture: wgpu::Texture,
    depth_view:    wgpu::TextureView,

    pipeline: OptimisedPipeline,

    /// Compute frustum cull pipeline.
    cull_pipeline:        crate::cull_pipeline::CullPipeline,
    cull_camera_bg:       wgpu::BindGroup,
    cull_origins_bg:      wgpu::BindGroup,
    cull_indirect_bg:     wgpu::BindGroup,

    /// Storage buffer of chunk world-space origins, one vec4 per pool slot.
    chunk_origins_buf:        wgpu::Buffer,
    chunk_origins_bind_group: wgpu::BindGroup,

    pub world: WorldManager,

    /// True after the first frame flushes pending chunk uploads.
    uploads_flushed: bool,

    /// True after the first frame logs the draw call summary.
    logged_draw_summary: bool,
}

impl OptimisedRenderer {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(
        window: Arc<Window>,
        width:  u32,
        height: u32,
        extent: WorldExtent,
    ) -> Result<Self, GpuError> {
        let instance = GpuContext::create_instance();
        let surface  = instance.create_surface(Arc::clone(&window)).expect("create surface");
        let gpu      = GpuContext::from_surface(instance, &surface, wgpu::Features::MULTI_DRAW_INDIRECT)?;
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
            usage:    TextureUsages::RENDER_ATTACHMENT,
            format:   surface_format,
            width:    width.max(1),
            height:   height.max(1),
            present_mode,
            alpha_mode:                    caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &config);

        let camera_buf = create_uniform_buffer::<CameraUniform>(&gpu.device, "camera uniform");

        let camera_bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("camera bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
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
            label:   Some("camera bg"),
            layout:  &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let (depth_texture, depth_view) =
            Self::create_depth_texture(&gpu.device, width.max(1), height.max(1));

        let pipeline = OptimisedPipeline::new(&gpu.device, surface_format, &camera_bgl);
        let world    = WorldManager::new(&gpu.device, &pipeline, extent);

        // Chunk origins storage buffer — one vec4 per pool slot.
        let origins_size = (crate::vertex_pool::MAX_SLOTS * 4 * 4) as u64;
        let chunk_origins_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("chunk origins storage"),
            size:               origins_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self::upload_origins(&gpu.queue, &chunk_origins_buf, &world);

        let chunk_origins_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("chunk origins bg"),
            layout:  &pipeline.chunk_origin_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: chunk_origins_buf.as_entire_binding(),
            }],
        });

        // ── Compute cull pipeline ─────────────────────────────────────────────
        let cull_pipeline = crate::cull_pipeline::CullPipeline::new(&gpu.device);

        let cull_camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("cull camera bg"),
            layout:  &cull_pipeline.camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let cull_origins_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("cull origins bg"),
            layout:  &cull_pipeline.origins_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: chunk_origins_buf.as_entire_binding(),
            }],
        });

        let cull_indirect_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("cull indirect bg"),
            layout:  &cull_pipeline.indirect_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: world.indirect_buffer.buffer.as_entire_binding(),
            }],
        });

        Ok(OptimisedRenderer {
            window, gpu, surface, config,
            camera_buf, camera_bind_group, camera_bgl,
            depth_texture, depth_view,
            pipeline,
            cull_pipeline, cull_camera_bg, cull_origins_bg, cull_indirect_bg,
            chunk_origins_buf, chunk_origins_bind_group,
            world,
            uploads_flushed:     false,
            logged_draw_summary: false,
        })
    }

    // ── Brush pass-throughs ───────────────────────────────────────────────────

    pub fn raycast(&self, camera: &Camera) -> Option<RayHit> {
        self.world.raycast(camera)
    }

    pub fn dig(&mut self, hit: &RayHit) {
        self.world.dig(&self.gpu.device, &self.gpu.queue, &self.pipeline, hit);
        self.refresh_origins();
    }

    pub fn place(&mut self, hit: &RayHit) {
        self.world.place(&self.gpu.device, &self.gpu.queue, &self.pipeline, hit);
        self.refresh_origins();
    }

    pub fn cycle_place_voxel(&mut self)  { self.world.cycle_place_voxel(); }
    pub fn increase_brush(&mut self)     { self.world.increase_brush(); }
    pub fn decrease_brush(&mut self)     { self.world.decrease_brush(); }
    pub fn save(&mut self)               { self.world.save(); }
    pub fn dirty_count(&self) -> usize   { self.world.dirty_count() }

    // ── Resize ────────────────────────────────────────────────────────────────

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width  = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        let (tex, view) = Self::create_depth_texture(&self.gpu.device, width, height);
        self.depth_texture = tex;
        self.depth_view    = view;
    }

    // ── Render ────────────────────────────────────────────────────────────────

    pub fn render(&mut self, camera: &Camera) -> Result<(), wgpu::SurfaceError> {
        if !self.uploads_flushed {
            self.world.flush_pending_uploads(
                &self.gpu.device, &self.gpu.queue, &self.pipeline,
            );
            Self::upload_origins(&self.gpu.queue, &self.chunk_origins_buf, &self.world);
            self.uploads_flushed = true;
        }

        write_uniform(
            &self.gpu.queue,
            &self.camera_buf,
            &CameraUniform::from_camera(camera),
        );

        let output = self.surface.get_current_texture()?;
        let view   = output.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = self.gpu.device.create_command_encoder(
            &CommandEncoderDescriptor { label: Some("optimised frame") },
        );

        let draw_count = self.world.indirect_buffer.draw_count;

        const CULL_MIN_DRAWS: u32 = 50;
        if draw_count >= CULL_MIN_DRAWS {
            self.world.indirect_buffer.reset_instance_counts(&self.gpu.queue);
            {
                let mut cpass = encoder.begin_compute_pass(
                    &wgpu::ComputePassDescriptor {
                        label: Some("frustum cull pass"),
                        timestamp_writes: None,
                    }
                );
                cpass.set_pipeline(&self.cull_pipeline.pipeline);
                cpass.set_bind_group(0, &self.cull_camera_bg,   &[]);
                cpass.set_bind_group(1, &self.cull_origins_bg,  &[]);
                cpass.set_bind_group(2, &self.cull_indirect_bg, &[]);
                let groups = crate::cull_pipeline::CullPipeline::dispatch_size(draw_count);
                cpass.dispatch_workgroups(groups, 1, 1);
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("optimised pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load:  LoadOp::Clear(wgpu::Color { r: 0.53, g: 0.81, b: 0.98, a: 1.0 }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(Operations {
                        load:  LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes:    None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group,          &[]);
            pass.set_bind_group(1, &self.chunk_origins_bind_group,   &[]);
            pass.set_bind_group(2, &self.world.vertex_pool.bind_group, &[]);

            if draw_count > 0 {
                pass.multi_draw_indirect(
                    &self.world.indirect_buffer.buffer,
                    0,
                    draw_count,
                );
            }

            if !self.logged_draw_summary {
                log::info!(
                    "multi_draw_indirect: {} draws submitted, frustum cull active",
                    draw_count
                );
                self.logged_draw_summary = true;
            }
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    /// Uploads all chunk origins into the storage buffer indexed by slot number.
    fn upload_origins(queue: &wgpu::Queue, buf: &wgpu::Buffer, world: &WorldManager) {
        for (pos, record) in world.vertex_pool.chunks.iter() {
            let slot = record.slot_range.first;
            let origin: [f32; 4] = [
                (pos.x * 32) as f32,
                (pos.y * 32) as f32,
                (pos.z * 32) as f32,
                0.0,
            ];
            let byte_offset = (slot * 16) as u64;
            queue.write_buffer(buf, byte_offset, bytemuck::bytes_of(&origin));
        }
    }

    /// Called after any remesh to keep chunk_origins_buf in sync.
    pub fn refresh_origins(&self) {
        Self::upload_origins(&self.gpu.queue, &self.chunk_origins_buf, &self.world);
    }

    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32)
        -> (wgpu::Texture, wgpu::TextureView)
    {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("depth texture"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          Self::DEPTH_FORMAT,
            usage:           TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        (texture, view)
    }
}