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
use crate::world_manager::WorldManager;

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

    pub world: WorldManager,

    /// True after the first frame flushes pending chunk uploads.
    uploads_flushed: bool,

    /// True after the first frame logs the draw call summary.
    logged_draw_summary: bool,
}

impl OptimisedRenderer {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(window: Arc<Window>, width: u32, height: u32) -> Result<Self, GpuError> {
        let instance = GpuContext::create_instance();
        let surface  = instance.create_surface(Arc::clone(&window)).expect("create surface");
        let gpu      = GpuContext::from_surface(instance, &surface, wgpu::Features::empty())?;
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
        let world    = WorldManager::new(&gpu.device, &pipeline);

        Ok(OptimisedRenderer {
            window, gpu, surface, config,
            camera_buf, camera_bind_group, camera_bgl,
            depth_texture, depth_view,
            pipeline, world,
            uploads_flushed: false,
            logged_draw_summary: false,
        })
    }

    // ── Brush pass-throughs ───────────────────────────────────────────────────

    pub fn raycast(&self, camera: &Camera) -> Option<RayHit> {
        self.world.raycast(camera)
    }

    pub fn dig(&mut self, hit: &RayHit) {
        self.world.dig(&self.gpu.device, &self.gpu.queue, &self.pipeline, hit);
    }

    pub fn place(&mut self, hit: &RayHit) {
        self.world.place(&self.gpu.device, &self.gpu.queue, &self.pipeline, hit);
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
        // On the first frame, flush all pending chunk uploads into the pool.
        if !self.uploads_flushed {
            self.world.flush_pending_uploads(
                &self.gpu.device, &self.gpu.queue, &self.pipeline,
            );
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
                timestamp_writes:  None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(2, &self.world.vertex_pool.bind_group, &[]);

            let mut drawn = 0u32;
            let mut skipped_zero = 0u32;
            for (pos, draw) in self.world.chunk_draws.iter() {
                if draw.vertex_count == 0 {
                    skipped_zero += 1;
                    log::warn!("chunk {:?} has vertex_count=0 — not drawn (upload failed?)", pos);
                    continue;
                }
                pass.set_bind_group(1, &draw.origin_bind_group, &[]);
                pass.draw(0..draw.vertex_count, draw.first_quad..draw.first_quad + 1);
                drawn += 1;
            }
            if !self.logged_draw_summary {
                log::info!(
                    "draw summary: {} chunks drawn, {} skipped (vertex_count=0) of {} total",
                    drawn, skipped_zero, self.world.chunk_draws.len()
                );
                self.logged_draw_summary = true;
            }
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    #[allow(dead_code)]
    pub fn surface_format(&self) -> wgpu::TextureFormat { self.config.format }

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