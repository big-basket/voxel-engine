/// Headless benchmark runner.
///
/// Called from main() when `--bench` is passed on the command line.
/// Loads scene configuration from `benchmark_config.json` in the workspace
/// root, falling back to hardcoded defaults if the file is absent.

use wgpu::util::DeviceExt;

use voxel_core::{
    benchmark::{BenchmarkConfig, MetricsCollector, Recorder, SceneKind},
    camera::{Camera, CameraUniform},
    gen::generate_chunk,
    gpu::{GpuContext, GpuError, write_uniform},
};

use crate::mesh::build_chunk_mesh;
use crate::pipeline::{ChunkUniform, NaivePipeline};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const WIDTH:  u32 = 1280;
const HEIGHT: u32 = 720;
const CONFIG_PATH: &str = "benchmark_config.json";

pub fn run_benchmarks() {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info,wgpu_core=warn,wgpu_hal=warn,naga=warn");
    }

    log::info!("=== Naive renderer benchmark ===");

    let config_path = std::path::Path::new(CONFIG_PATH);
    if !config_path.exists() {
        log::info!("Writing default benchmark_config.json — edit to customise scenes.");
        if let Err(e) = BenchmarkConfig::write_default(config_path) {
            log::warn!("Could not write default config: {e}");
        }
    }
    let config = BenchmarkConfig::load_or_default(config_path);
    log::info!("Running {} scene(s)", config.scenes.len());

    let gpu = match GpuContext::new_headless(wgpu::Features::empty()) {
        Ok(g) => g,
        Err(GpuError::NoAdapter) => {
            eprintln!("No GPU adapter found — cannot run benchmarks.");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("GPU init error: {e}");
            std::process::exit(1);
        }
    };
    log::info!("GPU: {}", gpu.adapter_info());

    let camera_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench camera uniform"),
        size: CameraUniform::SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let camera_bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bench camera bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(CameraUniform::SIZE),
            },
            count: None,
        }],
    });

    let camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bench camera bg"),
        layout: &camera_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buf.as_entire_binding(),
        }],
    });

    let pipeline = NaivePipeline::new(&gpu.device, RENDER_FORMAT, &camera_bgl, DEPTH_FORMAT);

    let (_render_tex, render_view) = make_render_target(&gpu.device);
    let (_depth_tex, depth_view) = make_depth_target(&gpu.device);

    let recorder = Recorder::new("naive", "results");

    for scene in &config.scenes {
        log::info!("--- Scene: {} ---", scene.id);
        log::info!("  {}", scene.description);

        let mut camera = Camera::new(WIDTH as f32 / HEIGHT as f32);
        camera.position = scene.camera_pos();
        camera.forward  = scene.camera_forward();

        match &scene.kind {
            SceneKind::StaticHighDensity { draw_radius, vertical_layers } => {
                let chunk_draws = build_scene_draws(
                    &gpu.device, &pipeline, &scene.terrain,
                    *draw_radius, *vertical_layers,
                );
                run_static_scene(
                    &gpu, &pipeline, &camera_buf, &camera_bg,
                    &render_view, &depth_view,
                    &chunk_draws, &camera,
                    scene.warmup_frames, scene.measure_frames,
                    &recorder, &scene.id, &scene.description,
                );
            }

            SceneKind::DynamicRemesh { .. } => {
                let chunk_draws = build_scene_draws(
                    &gpu.device, &pipeline, &scene.terrain, 6, 3,
                );
                run_static_scene(
                    &gpu, &pipeline, &camera_buf, &camera_bg,
                    &render_view, &depth_view,
                    &chunk_draws, &camera,
                    scene.warmup_frames, scene.measure_frames,
                    &recorder, &scene.id, &scene.description,
                );
            }

            SceneKind::StressTest { voxels_per_step, fps_floor } => {
                run_stress_test(
                    &gpu, &pipeline, &camera_buf, &camera_bg,
                    &render_view, &depth_view,
                    &camera, &scene.terrain,
                    scene.warmup_frames, scene.measure_frames,
                    *voxels_per_step, *fps_floor,
                    &recorder, &scene.id, &scene.description,
                );
            }
        }
    }

    log::info!("Benchmark complete. Results written to results/");
}

// ── Scene runners ─────────────────────────────────────────────────────────────

fn render_frame(
    gpu: &GpuContext,
    pipeline: &NaivePipeline,
    camera_buf: &wgpu::Buffer,
    camera_bg: &wgpu::BindGroup,
    render_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    chunk_draws: &[DrawCall],
    camera: &Camera,
    mut collector: Option<&mut MetricsCollector>,
) {
    let uniform = CameraUniform::from_camera(camera);
    write_uniform(&gpu.queue, camera_buf, &uniform);

    let mut encoder = gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("bench frame") },
    );
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bench pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: render_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.53, g: 0.81, b: 0.98, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, camera_bg, &[]);
        for draw in chunk_draws {
            pass.set_bind_group(1, &draw.chunk_bind_group, &[]);
            pass.set_vertex_buffer(0, draw.vertex_buf.slice(..));
            pass.set_index_buffer(draw.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..draw.index_count, 0, 0..1);
            if let Some(c) = collector.as_mut() {
                c.record_draw(draw.vertex_count, draw.index_count as u64);
            }
        }
    }
    gpu.queue.submit(std::iter::once(encoder.finish()));
    gpu.device.poll(wgpu::Maintain::Wait);
}

#[allow(clippy::too_many_arguments)]
fn run_static_scene(
    gpu: &GpuContext,
    pipeline: &NaivePipeline,
    camera_buf: &wgpu::Buffer,
    camera_bg: &wgpu::BindGroup,
    render_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    chunk_draws: &[DrawCall],
    camera: &Camera,
    warmup_frames: u32,
    measure_frames: u32,
    recorder: &Recorder,
    scene_id: &str,
    scene_description: &str,
) {
    let mut collector = MetricsCollector::new();

    for frame_idx in 0..(warmup_frames + measure_frames) {
        let measuring = frame_idx >= warmup_frames;
        if measuring { collector.begin_frame(); }
        render_frame(gpu, pipeline, camera_buf, camera_bg, render_view, depth_view,
            chunk_draws, camera,
            if measuring { Some(&mut collector) } else { None },
        );
        if measuring { collector.end_frame(0); }
    }

    let summary = collector.summarise();
    log::info!(
        "  avg_fps={:.1}  1%_low={:.1}  avg_ms={:.2}  triangles={}  draws={}",
        summary.avg_fps, summary.one_pct_low_fps, summary.avg_frame_ms,
        summary.avg_triangle_count, summary.avg_draw_calls,
    );
    if let Err(e) = recorder.write_all(scene_id, scene_description, collector.frames(), &summary) {
        log::error!("Failed to write results for {scene_id}: {e}");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_stress_test(
    gpu: &GpuContext,
    pipeline: &NaivePipeline,
    camera_buf: &wgpu::Buffer,
    camera_bg: &wgpu::BindGroup,
    render_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    camera: &Camera,
    terrain: &voxel_core::gen::TerrainParams,
    warmup_frames: u32,
    max_frames: u32,
    _voxels_per_step: u32,
    fps_floor: f32,
    recorder: &Recorder,
    scene_id: &str,
    scene_description: &str,
) {
    // Stress test: start at radius 1, expand by 1 chunk ring every 60 frames.
    // Records each step's FPS until the floor is hit.
    let mut collector = MetricsCollector::new();
    let mut current_radius = 1i32;
    let mut chunk_draws = build_scene_draws(&gpu.device, pipeline, terrain, current_radius, 2);
    let frames_per_step = 60u32;
    let mut frames_at_current_radius = 0u32;
    let max_radius = 16i32;

    // Warmup at radius 1
    for _ in 0..warmup_frames {
        render_frame(gpu, pipeline, camera_buf, camera_bg, render_view, depth_view,
            &chunk_draws, camera, None);
    }

    for _ in 0..max_frames {
        collector.begin_frame();
        render_frame(gpu, pipeline, camera_buf, camera_bg, render_view, depth_view,
            &chunk_draws, camera, Some(&mut collector));
        collector.end_frame(0);

        frames_at_current_radius += 1;

        // Check rolling FPS every step interval
        if frames_at_current_radius >= frames_per_step {
            let recent: Vec<f64> = collector.frames()
                .iter().rev().take(frames_per_step as usize)
                .map(|f| f.fps).collect();
            let rolling_fps = recent.iter().sum::<f64>() / recent.len() as f64;

            log::info!(
                "  radius={:2}  chunks={}  avg_fps={:.1}",
                current_radius,
                chunk_draws.len(),
                rolling_fps,
            );

            if rolling_fps < fps_floor as f64 {
                log::info!("  FPS floor ({fps_floor}) reached at radius {current_radius}");
                break;
            }

            // Expand to next radius
            current_radius += 1;
            if current_radius > max_radius {
                log::info!("  Reached max radius {max_radius} without hitting FPS floor");
                break;
            }
            chunk_draws = build_scene_draws(&gpu.device, pipeline, terrain, current_radius, 2);
            frames_at_current_radius = 0;
        }
    }

    let summary = collector.summarise();
    log::info!(
        "  final: avg_fps={:.1}  1%_low={:.1}  max_radius={}  triangles={}",
        summary.avg_fps, summary.one_pct_low_fps, current_radius,
        summary.avg_triangle_count,
    );
    if let Err(e) = recorder.write_all(scene_id, scene_description, collector.frames(), &summary) {
        log::error!("Failed to write results for {scene_id}: {e}");
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

struct DrawCall {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    vertex_count: u64,
    _chunk_buf: wgpu::Buffer,
    chunk_bind_group: wgpu::BindGroup,
}

fn build_scene_draws(
    device: &wgpu::Device,
    pipeline: &NaivePipeline,
    terrain: &voxel_core::gen::TerrainParams,
    draw_radius: i32,
    vertical_layers: i32,
) -> Vec<DrawCall> {
    use voxel_core::world::{World, CHUNK_SIZE_I};
    use glam::IVec3;

    let r = draw_radius;
    let x_range = -r..=r;
    let y_range = -vertical_layers..=0;
    let z_range = -r..=r;

    log::info!(
        "  Building world: {}×{}×{} chunks ({} total)...",
        r * 2 + 1, vertical_layers + 1, r * 2 + 1,
        (r * 2 + 1) * (vertical_layers + 1) * (r * 2 + 1)
    );

    let mut world = World::new();
    for cy in y_range.clone() {
        for cz in z_range.clone() {
            for cx in x_range.clone() {
                let pos = IVec3::new(cx, cy, cz);
                world.insert_chunk(pos, generate_chunk(pos, terrain));
            }
        }
    }

    let mut draws = Vec::new();
    for cy in y_range {
        for cz in z_range.clone() {
            for cx in x_range.clone() {
                let chunk_pos = IVec3::new(cx, cy, cz);
                let chunk = world.get_chunk(&chunk_pos).unwrap();
                let (verts, idx) = build_chunk_mesh(chunk, chunk_pos, &world);
                if verts.is_empty() { continue; }

                let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bench vbuf"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bench ibuf"),
                    contents: bytemuck::cast_slice(&idx),
                    usage: wgpu::BufferUsages::INDEX,
                });

                let origin = [
                    (cx * CHUNK_SIZE_I) as f32,
                    (cy * CHUNK_SIZE_I) as f32,
                    (cz * CHUNK_SIZE_I) as f32,
                    0.0f32,
                ];
                let chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bench chunk uniform"),
                    contents: bytemuck::bytes_of(&ChunkUniform { origin }),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let chunk_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bench chunk bg"),
                    layout: &pipeline.chunk_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: chunk_buf.as_entire_binding(),
                    }],
                });

                draws.push(DrawCall {
                    vertex_count: verts.len() as u64,
                    index_count: idx.len() as u32,
                    vertex_buf,
                    index_buf,
                    _chunk_buf: chunk_buf,
                    chunk_bind_group,
                });
            }
        }
    }

    log::info!("  Meshed {} non-empty chunks into {} draw calls", 
        (r * 2 + 1) * (vertical_layers + 1) * (r * 2 + 1),
        draws.len()
    );
    draws
}

fn make_render_target(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench render target"),
        size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RENDER_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_depth_target(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench depth"),
        size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}