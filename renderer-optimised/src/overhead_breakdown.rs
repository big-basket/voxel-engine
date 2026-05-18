/// Overhead breakdown benchmark

use glam::IVec3;
use std::time::{Duration, Instant};
use std::io::Write;

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
const COUNTS: &[i32] = &[1, 4, 9, 16, 36, 64, 100, 225, 400];

pub fn run_overhead_breakdown() {
    log::info!("=== Overhead breakdown benchmark ===");

    let gpu = match GpuContext::new_headless(wgpu::Features::MULTI_DRAW_INDIRECT) {
        Ok(g)  => g,
        Err(e) => { eprintln!("GPU error: {e}"); std::process::exit(1); }
    };
    log::info!("GPU: {}", gpu.adapter_info());

    // Shared GPU resources 
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
    write_uniform(&gpu.queue, &camera_buf, &CameraUniform::from_camera(&camera));

    // Naive pipeline for comparison draw calls
    let naive_pipeline = crate::pipeline::OptimisedPipeline::new(
        &gpu.device, RENDER_FORMAT, &camera_bgl,
    );

    // Print header 

    println!("\n{:>6} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} {:>10} | {:>10}",
        "chunks", "opt_total", "cull_reset", "cull_disp",
        "naive_us", "upload_ms", "vram_mib", "quads", "note");
    println!("{}", "-".repeat(120));

    struct Row {
        chunks: usize,
        naive_us: f64,
        opt_us: f64,
    }
    let mut csv_rows: Vec<Row> = Vec::new();

    // Per-chunk-count loop 
    for &n in COUNTS {
        let side = (n as f64).sqrt().ceil() as i32;
        let actual_n = (side * side) as usize;

        // Build world
        let mut world = World::new();
        for cz in 0..side { for cx in 0..side {
            let pos = IVec3::new(cx - side/2, 0, cz - side/2);
            world.insert_chunk(pos, generate_chunk(pos, &terrain));
        }}

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

        let total_quads = pool.total_quads();
        let vram_mib    = pool.allocated_bytes() as f64 / (1024.0 * 1024.0);

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

        // indirect buffer reset 
        let mut reset_times = Vec::with_capacity(FRAMES as usize);
        for i in 0..WARMUP + FRAMES {
            let t = Instant::now();
            indirect.reset_instance_counts(&gpu.queue);
            gpu.device.poll(wgpu::Maintain::Wait);
            if i >= WARMUP { reset_times.push(t.elapsed()); }
        }
        let reset_avg_us = avg_us(&reset_times);

        // compute cull dispatch 
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

        // full optimised frame 
        let mut opt_times = Vec::with_capacity(FRAMES as usize);
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
                rp.set_bind_group(0, &camera_bg,         &[]);
                rp.set_bind_group(1, &origins_bg_render, &[]);
                rp.set_bind_group(2, &quad_bg,           &[]);
                if draw_count > 0 {
                    rp.multi_draw_indirect(&indirect.buffer, 0, draw_count);
                }
            }
            gpu.queue.submit(std::iter::once(enc.finish()));
            gpu.device.poll(wgpu::Maintain::Wait);
            if i >= WARMUP { opt_times.push(t.elapsed()); }
        }
        let opt_total_us = avg_us(&opt_times);

        // naive equivalent 
        let mut naive_times = Vec::with_capacity(FRAMES as usize);
        for i in 0..WARMUP + FRAMES {
            let t = Instant::now();
            let mut enc = gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("naive frame") });
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("naive render"),
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
                rp.set_pipeline(&naive_pipeline.pipeline);
                rp.set_bind_group(0, &camera_bg, &[]);
                rp.set_bind_group(2, &quad_bg,   &[]);
          
                for entry_idx in 0..draw_count {
                    let offset = (entry_idx as u64)
                        * std::mem::size_of::<DrawIndirectArgs>() as u64;
                    rp.set_bind_group(1, &origins_bg_render, &[]);
                    rp.draw_indirect(&indirect.buffer, offset);
                }
            }
            gpu.queue.submit(std::iter::once(enc.finish()));
            gpu.device.poll(wgpu::Maintain::Wait);
            if i >= WARMUP { naive_times.push(t.elapsed()); }
        }
        let naive_total_us = avg_us(&naive_times);

        let note = if actual_n != n as usize {
            format!("actual={}", actual_n)
        } else {
            String::new()
        };

        println!("{:>6} | {:>9.1}µs {:>9.1}µs {:>9.1}µs | {:>9.1}µs {:>9.1}ms {:>8.1} {:>8} | {}",
            draw_count,
            opt_total_us, reset_avg_us, cull_avg_us,
            naive_total_us, upload_ms, vram_mib, total_quads,
            note
        );

        csv_rows.push(Row {
            chunks: draw_count as usize,
            naive_us: naive_total_us,
            opt_us: opt_total_us,
        });
    }

    println!("\nLegend:");
    println!("  opt_total  = full optimised frame (reset + cull dispatch + multi_draw)");
    println!("  cull_reset = CPU write_buffer to restore instance_count=1");
    println!("  cull_disp  = compute dispatch time inc. GPU sync");
    println!("  naive_us   = N individual draw_indirect calls (naive O(N) equivalent)");
    println!("  upload_ms  = one-time mesh upload time (not part of frame time)");
    println!("  vram_mib   = vertex pool VRAM (allocated_slots × 32 KiB)");
    println!("  quads      = total greedy quads in pool");



    let csv_path = std::path::Path::new("results/overhead.csv");
    std::fs::create_dir_all("results").ok();
    match std::fs::File::create(csv_path) {
        Ok(mut f) => {
            writeln!(f, "chunks,naive_us,opt_us").ok();
            for row in &csv_rows {
                writeln!(f, "{},{:.1},{:.1}", row.chunks, row.naive_us, row.opt_us).ok();
            }
            println!("\nWrote overhead data to {}", csv_path.display());
            println!("Run: python3 scripts/plot_benchmarks.py to generate Fig 12.");
        }
        Err(e) => eprintln!("\nWarning: could not write overhead.csv: {e}"),
    }
}

fn avg_us(times: &[Duration]) -> f64 {
    if times.is_empty() { return 0.0; }
    times.iter().map(|d| d.as_secs_f64() * 1_000_000.0).sum::<f64>() / times.len() as f64
}