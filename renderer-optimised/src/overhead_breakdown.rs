/// Overhead breakdown benchmark.
///
/// Measures each renderer stage independently to identify where the
/// optimised renderer loses to the naive one at low chunk counts, and
/// find the crossover point where it wins.
///
/// Stages measured:
///   naive:     camera write | draw loop (N × draw_indexed) | submit
///   optimised: camera write | indirect reset | cull dispatch | multi_draw | submit
///
/// Run with: cargo run --release -p renderer-optimised -- --overhead

use glam::IVec3;
use std::time::{Duration, Instant};

use voxel_core::{
    camera::{Camera, CameraUniform},
    gen::{TerrainParams, generate_chunk, mesh_chunk},
    gpu::{GpuContext, GpuError, write_uniform},
    world::{World, CHUNK_SIZE_I},
    benchmark::{RENDER_FORMAT, DEPTH_FORMAT, make_render_target, make_depth_target},
};

use crate::cull_pipeline::CullPipeline;
use crate::indirect::IndirectBuffer;
use crate::pipeline::OptimisedPipeline;
use crate::vertex_pool::{DrawIndirectArgs, VertexPool, MAX_SLOTS};

const WARMUP: u32  = 50;
const FRAMES: u32  = 200;
// Chunk counts to test — sweeps from tiny to large to find the crossover point
const COUNTS: &[i32] = &[1, 2, 4, 8, 16, 32, 64, 100, 200, 400];

pub fn run_overhead_breakdown() {
    log::info!("=== Overhead breakdown benchmark ===");

    let gpu = match GpuContext::new_headless(wgpu::Features::MULTI_DRAW_INDIRECT) {
        Ok(g)  => g,
        Err(e) => { eprintln!("GPU error: {e}"); std::process::exit(1); }
    };
    log::info!("GPU: {}", gpu.adapter_info());

    let camera_bgl = gpu.device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("ob camera bgl"),
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

    let camera_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ob camera buf"),
        size: CameraUniform::SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ob camera bg"),
        layout: &camera_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0, resource: camera_buf.as_entire_binding(),
        }],
    });

    let pipeline      = OptimisedPipeline::new(&gpu.device, RENDER_FORMAT, &camera_bgl);
    let cull_pipeline = CullPipeline::new(&gpu.device);

    let origins_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ob origins"),
        size: (MAX_SLOTS * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let origins_bg_render = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ob origins bg render"),
        layout: &pipeline.chunk_origin_bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: origins_buf.as_entire_binding() }],
    });
    let cull_origins_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ob cull origins bg"),
        layout: &cull_pipeline.origins_bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: origins_buf.as_entire_binding() }],
    });
    let cull_camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ob cull camera bg"),
        layout: &cull_pipeline.camera_bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() }],
    });

    let (_rt, render_view) = make_render_target(&gpu.device);
    let (_dt, depth_view)  = make_depth_target(&gpu.device);

    let terrain = TerrainParams::default();

    let mut camera = Camera::new(1280.0 / 720.0);
    camera.position = glam::Vec3::new(0.0, 120.0, -200.0);
    camera.forward  = glam::Vec3::new(0.0, -0.25, 1.0).normalize();

    println!("\n{:>6} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} {:>10} | {:>10}",
        "chunks", "opt_total", "cull_reset", "cull_disp",
        "indirect_rb", "upload_ms", "vram_mib", "quads", "note");
    println!("{}", "-".repeat(120));

    for &n in COUNTS {
        // Build a square patch of n chunks at the surface layer.
        let side    = (n as f64).sqrt().ceil() as i32;
        let mut world = World::new();
        let params  = terrain.clone();

        for cz in 0..side { for cx in 0..side {
            let pos = IVec3::new(cx - side/2, 0, cz - side/2);
            world.insert_chunk(pos, generate_chunk(pos, &params));
        }}

        let actual_n = (side * side) as usize;

        // Build vertex pool
        let t_upload = Instant::now();
        let mut pool = VertexPool::new(&gpu.device);
        for cz in 0..side { for cx in 0..side {
            let pos   = IVec3::new(cx - side/2, 0, cz - side/2);
            let chunk = world.get_chunk(&pos).unwrap();
            let quads = mesh_chunk(chunk, None);
            if !quads.is_empty() {
                pool.upload_chunk(&gpu.device, &gpu.queue, pos, &quads);
            }
        }}
        gpu.device.poll(wgpu::Maintain::Wait);
        let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;

        let total_quads   = pool.total_quads();
        let vram_bytes    = pool.allocated_bytes();
        let vram_mib      = vram_bytes as f64 / (1024.0 * 1024.0);

        // Upload chunk origins
        for (pos, record) in pool.chunks.iter() {
            let slot   = record.slot_range.first;
            let origin = [
                (pos.x * CHUNK_SIZE_I) as f32,
                (pos.y * CHUNK_SIZE_I) as f32,
                (pos.z * CHUNK_SIZE_I) as f32,
                0.0f32,
            ];
            gpu.queue.write_buffer(&origins_buf, (slot * 16) as u64, bytemuck::bytes_of(&origin));
        }

        // Build indirect buffer
        let active: Vec<(IVec3, DrawIndirectArgs)> = pool.active_draw_args().collect();
        let mut indirect = IndirectBuffer::new(&gpu.device, MAX_SLOTS as u32);
        indirect.rebuild(&gpu.queue, &active);
        let draw_count = indirect.draw_count;

        let quad_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ob quad bg"),
            layout: &pipeline.quad_storage_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: pool.quad_buffer.as_entire_binding() }],
        });
        let cull_indirect_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ob cull indirect bg"),
            layout: &cull_pipeline.indirect_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: indirect.buffer.as_entire_binding() }],
        });

        // ── Measure: indirect buffer reset (CPU write_buffer) ─────────────────
        let mut reset_times = Vec::with_capacity(FRAMES as usize);
        for i in 0..WARMUP + FRAMES {
            let t = Instant::now();
            indirect.reset_instance_counts(&gpu.queue);
            gpu.device.poll(wgpu::Maintain::Wait);
            if i >= WARMUP { reset_times.push(t.elapsed()); }
        }
        let reset_avg_us = avg_us(&reset_times);

        // ── Measure: compute cull dispatch ────────────────────────────────────
        let mut cull_times = Vec::with_capacity(FRAMES as usize);
        for i in 0..WARMUP + FRAMES {
            indirect.reset_instance_counts(&gpu.queue);
            let t = Instant::now();
            let mut enc = gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("cull bench") });
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cull"), timestamp_writes: None });
                cp.set_pipeline(&cull_pipeline.pipeline);
                cp.set_bind_group(0, &cull_camera_bg, &[]);
                cp.set_bind_group(1, &cull_origins_bg, &[]);
                cp.set_bind_group(2, &cull_indirect_bg, &[]);
                cp.dispatch_workgroups(CullPipeline::dispatch_size(draw_count), 1, 1);
            }
            gpu.queue.submit(std::iter::once(enc.finish()));
            gpu.device.poll(wgpu::Maintain::Wait);
            if i >= WARMUP { cull_times.push(t.elapsed()); }
        }
        let cull_avg_us = avg_us(&cull_times);

        // ── Measure: full optimised frame (reset + cull + render) ─────────────
        write_uniform(&gpu.queue, &camera_buf, &CameraUniform::from_camera(&camera));
        let mut frame_times = Vec::with_capacity(FRAMES as usize);
        for i in 0..WARMUP + FRAMES {
            let t = Instant::now();
            indirect.reset_instance_counts(&gpu.queue);
            let mut enc = gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("opt frame") });
            if draw_count > 0 {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cull"), timestamp_writes: None });
                cp.set_pipeline(&cull_pipeline.pipeline);
                cp.set_bind_group(0, &cull_camera_bg, &[]);
                cp.set_bind_group(1, &cull_origins_bg, &[]);
                cp.set_bind_group(2, &cull_indirect_bg, &[]);
                cp.dispatch_workgroups(CullPipeline::dispatch_size(draw_count), 1, 1);
            }
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("opt render"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &render_view, resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None, occlusion_query_set: None,
                });
                rp.set_pipeline(&pipeline.pipeline);
                rp.set_bind_group(0, &camera_bg,        &[]);
                rp.set_bind_group(1, &origins_bg_render, &[]);
                rp.set_bind_group(2, &quad_bg,           &[]);
                if draw_count > 0 {
                    rp.multi_draw_indirect(&indirect.buffer, 0, draw_count);
                }
            }
            gpu.queue.submit(std::iter::once(enc.finish()));
            gpu.device.poll(wgpu::Maintain::Wait);
            if i >= WARMUP { frame_times.push(t.elapsed()); }
        }
        let total_avg_us = avg_us(&frame_times);
        let fps = 1_000_000.0 / total_avg_us;

        let note = if actual_n != n as usize {
            format!("actual={}", actual_n)
        } else {
            String::new()
        };

        println!("{:>6} | {:>9.1}µs {:>9.1}µs {:>9.1}µs | {:>10} {:>9.1}ms {:>8.1} {:>8} | {}",
            draw_count,
            total_avg_us, reset_avg_us, cull_avg_us,
            format!("N/A"), upload_ms, vram_mib, total_quads,
            note
        );
    }

    println!("\nLegend:");
    println!("  opt_total   = full optimised frame time (µs)");
    println!("  cull_reset  = CPU write_buffer to reset instance_counts (µs)");
    println!("  cull_disp   = compute dispatch time inc. GPU sync (µs)");
    println!("  upload_ms   = one-time mesh upload time");
    println!("  vram_mib    = vertex pool VRAM usage");
    println!("  quads       = total greedy quads in pool");
    println!("\nCompare opt_total against naive ~0.3ms (3478 FPS) to find crossover.");
}

fn avg_us(times: &[Duration]) -> f64 {
    if times.is_empty() { return 0.0; }
    times.iter().map(|d| d.as_secs_f64() * 1_000_000.0).sum::<f64>() / times.len() as f64
}