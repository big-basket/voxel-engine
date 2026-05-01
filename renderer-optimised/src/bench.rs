/// Optimised renderer benchmark shim — see voxel-core/src/benchmark/runner.rs
/// for the shared loop. This file only contains what is specific to the
/// optimised renderer: vertex pool, indirect buffer, compute cull dispatch.
use glam::IVec3;
use voxel_core::{
    benchmark::{
        BenchRenderer, BenchmarkScene, MetricsCollector,
        RENDER_FORMAT, DEPTH_FORMAT,
        make_render_target, make_depth_target, run_all_scenes,
    },
    camera::CameraUniform,
    gen::{generate_chunk, mesh_chunk},
    gpu::{GpuContext, GpuError},
    world::{World, CHUNK_SIZE_I},
};
use crate::cull_pipeline::CullPipeline;
use crate::indirect::IndirectBuffer;
use crate::pipeline::OptimisedPipeline;
use crate::vertex_pool::{DrawIndirectArgs, VertexPool, MAX_SLOTS};
use voxel_core::camera::Camera;

pub struct OptimisedScene {
    pool:             VertexPool,
    indirect:         IndirectBuffer,
    render_view:      wgpu::TextureView,
    depth_view:       wgpu::TextureView,
    quad_bg:          wgpu::BindGroup,
    origins_bg:       wgpu::BindGroup,
    cull_indirect_bg: wgpu::BindGroup,
    total_quads:      u64,
}

pub struct OptimisedBenchRenderer {
    pipeline:        OptimisedPipeline,
    cull_pipeline:   CullPipeline,
    camera_bgl:      wgpu::BindGroupLayout,
    origins_buf:     wgpu::Buffer,
    cull_origins_bg: wgpu::BindGroup,
}

impl OptimisedBenchRenderer {
    fn new(gpu: &GpuContext) -> Self {
        let camera_bgl = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("bench opt camera bgl"),
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
            }
        );
        let pipeline      = OptimisedPipeline::new(&gpu.device, RENDER_FORMAT, &camera_bgl);
        let cull_pipeline = CullPipeline::new(&gpu.device);
        let origins_buf   = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench chunk origins"),
            size: (MAX_SLOTS * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_origins_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bench cull origins bg"),
            layout: &cull_pipeline.origins_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0, resource: origins_buf.as_entire_binding(),
            }],
        });
        OptimisedBenchRenderer {
            pipeline, cull_pipeline, camera_bgl, origins_buf, cull_origins_bg,
        }
    }
}

impl BenchRenderer for OptimisedBenchRenderer {
    type Scene = OptimisedScene;
    fn name(&self)        -> &str { "optimised" }
    fn results_dir(&self) -> &str { "results_optimised" }

    fn setup_scene(&mut self, gpu: &GpuContext, scene: &BenchmarkScene) -> OptimisedScene {
        let x_range = -2i32..=2;
        let y_range = -2i32..=1;
        let z_range = -2i32..=2;
        let mut world = World::new();
        for cy in y_range.clone() { for cz in z_range.clone() { for cx in x_range.clone() {
            let pos = IVec3::new(cx, cy, cz);
            world.insert_chunk(pos, generate_chunk(pos, &scene.terrain));
        }}}
        let mut pool = VertexPool::new(&gpu.device);
        for cy in y_range { for cz in z_range.clone() { for cx in x_range.clone() {
            let pos = IVec3::new(cx, cy, cz);
            let quads = mesh_chunk(world.get_chunk(&pos).unwrap(), None);
            if !quads.is_empty() { pool.upload_chunk(&gpu.device, &gpu.queue, pos, &quads); }
        }}}
        for (pos, record) in pool.chunks.iter() {
            let origin: [f32; 4] = [
                (pos.x * CHUNK_SIZE_I) as f32, (pos.y * CHUNK_SIZE_I) as f32,
                (pos.z * CHUNK_SIZE_I) as f32, 0.0,
            ];
            gpu.queue.write_buffer(&self.origins_buf, (record.slot_range.first * 16) as u64,
                bytemuck::bytes_of(&origin));
        }
        let active: Vec<(IVec3, DrawIndirectArgs)> = pool.active_draw_args().collect();
        let mut indirect = IndirectBuffer::new(&gpu.device, MAX_SLOTS as u32);
        indirect.rebuild(&gpu.queue, &active);
        let total_quads = pool.total_quads();
        let (_rt, render_view) = make_render_target(&gpu.device);
        let (_dt, depth_view)  = make_depth_target(&gpu.device);
        let quad_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bench quad bg"), layout: &self.pipeline.quad_storage_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: pool.quad_buffer.as_entire_binding() }],
        });
        let origins_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bench origins bg"), layout: &self.pipeline.chunk_origin_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.origins_buf.as_entire_binding() }],
        });
        let cull_indirect_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bench cull indirect bg"), layout: &self.cull_pipeline.indirect_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: indirect.buffer.as_entire_binding() }],
        });
        OptimisedScene { pool, indirect, render_view, depth_view, quad_bg, origins_bg, cull_indirect_bg, total_quads }
    }

    fn render_frame(&mut self, gpu: &GpuContext, s: &mut OptimisedScene,
        _camera: &Camera, camera_buf: &wgpu::Buffer, collector: &mut MetricsCollector)
    {
        let camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bench opt cam bg"), layout: &self.camera_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() }],
        });
        let cull_camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bench cull cam bg"), layout: &self.cull_pipeline.camera_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() }],
        });
        let draw_count = s.indirect.draw_count;
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("bench opt frame") });
        if draw_count > 0 {
            s.indirect.reset_instance_counts(&gpu.queue);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bench cull"), timestamp_writes: None });
            cpass.set_pipeline(&self.cull_pipeline.pipeline);
            cpass.set_bind_group(0, &cull_camera_bg, &[]);
            cpass.set_bind_group(1, &self.cull_origins_bg, &[]);
            cpass.set_bind_group(2, &s.cull_indirect_bg, &[]);
            cpass.dispatch_workgroups(CullPipeline::dispatch_size(draw_count), 1, 1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bench opt pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &s.render_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.53, g: 0.81, b: 0.98, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &s.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None, occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &camera_bg,    &[]);
            pass.set_bind_group(1, &s.origins_bg, &[]);
            pass.set_bind_group(2, &s.quad_bg,    &[]);
            if draw_count > 0 {
                pass.multi_draw_indirect(&s.indirect.buffer, 0, draw_count);
            }
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));
        collector.record_draw(s.total_quads * 6, s.total_quads * 6);
    }
}

pub fn run_benchmarks() {
    log::info!("=== Optimised renderer benchmark ===");
    let gpu = match GpuContext::new_headless(wgpu::Features::MULTI_DRAW_INDIRECT) {
        Ok(g)  => g,
        Err(GpuError::NoAdapter) => { eprintln!("No GPU adapter."); std::process::exit(1); }
        Err(e) => { eprintln!("GPU error: {e}"); std::process::exit(1); }
    };
    log::info!("GPU: {}", gpu.adapter_info());
    let camera_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench camera"), size: CameraUniform::SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut renderer = OptimisedBenchRenderer::new(&gpu);
    run_all_scenes(&mut renderer, &gpu, &camera_buf);
}