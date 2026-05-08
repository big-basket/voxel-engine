/// Naive renderer benchmark shim.
///
/// Implements `BenchRenderer` from `voxel_core::benchmark::runner` so the
/// shared `run_all_scenes` loop handles all frame timing, scene iteration,
/// and result writing. This file only contains what is unique to the naive
/// renderer: per-chunk VBO allocation and the N-draw-call render pass.

use wgpu::util::DeviceExt;

use voxel_core::{
    benchmark::{
        BenchRenderer, BenchmarkScene, MetricsCollector,
        RENDER_FORMAT, DEPTH_FORMAT,
        make_render_target, make_depth_target, run_all_scenes,
        scenes::SceneKind,
    },
    camera::{Camera, CameraUniform},
    gen::generate_chunk,
    gpu::{GpuContext, GpuError},
    world::{World, CHUNK_SIZE_I},
};
use glam::IVec3;

use crate::mesh::build_chunk_mesh;
use crate::pipeline::{ChunkUniform, NaivePipeline};

// ── Per-scene GPU state ───────────────────────────────────────────────────────

pub struct NaiveScene {
    draws:       Vec<ChunkDrawCall>,
    render_view: wgpu::TextureView,
    depth_view:  wgpu::TextureView,
}

struct ChunkDrawCall {
    vertex_buf:       wgpu::Buffer,
    index_buf:        wgpu::Buffer,
    index_count:      u32,
    vertex_count:     u64,
    chunk_bind_group: wgpu::BindGroup,
}

// ── BenchRenderer impl ────────────────────────────────────────────────────────

pub struct NaiveBenchRenderer {
    pipeline:   NaivePipeline,
    camera_bgl: wgpu::BindGroupLayout,
}

impl NaiveBenchRenderer {
    fn new(gpu: &GpuContext) -> Self {
        let camera_bgl = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("bench naive camera bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(CameraUniform::SIZE),
                    },
                    count: None,
                }],
            }
        );
        let pipeline = NaivePipeline::new(
            &gpu.device, RENDER_FORMAT, &camera_bgl, DEPTH_FORMAT,
        );
        NaiveBenchRenderer { pipeline, camera_bgl }
    }
}

impl BenchRenderer for NaiveBenchRenderer {
    type Scene = NaiveScene;

    fn name(&self)        -> &str { "naive" }
    fn results_dir(&self) -> &str { "results" }

    fn setup_scene(&mut self, gpu: &GpuContext, scene: &BenchmarkScene) -> NaiveScene {
        // Derive chunk ranges from the scene kind, matching world_manager.rs logic.
        let (draw_radius, vertical_layers) = match scene.kind {
            SceneKind::StaticHighDensity { draw_radius, vertical_layers } => {
                (draw_radius, vertical_layers)
            }
            _ => (8, 4),
        };

        // Anchor vertical range on sea_level so surface chunks are always loaded.
        let sea_chunk = (scene.terrain.sea_level as i32).div_euclid(32);
        let half_v    = vertical_layers / 2;
        let cy_min    = sea_chunk - half_v;
        let cy_max    = sea_chunk + half_v - 1;

        let x_range = -draw_radius..=draw_radius;
        let y_range = cy_min..=cy_max;
        let z_range = -draw_radius..=draw_radius;

        log::info!(
            "setup_scene '{}': draw_radius={} vertical_layers={} cy={}..={} \
             → {}×{}×{} = {} chunks",
            scene.id, draw_radius, vertical_layers, cy_min, cy_max,
            draw_radius * 2 + 1, vertical_layers, draw_radius * 2 + 1,
            (draw_radius * 2 + 1).pow(2) * vertical_layers,
        );

        // Generate world using the scene's actual terrain params.
        let mut world = World::new();
        for cy in y_range.clone() {
            for cz in z_range.clone() {
                for cx in x_range.clone() {
                    let pos = IVec3::new(cx, cy, cz);
                    world.insert_chunk(pos, generate_chunk(pos, &scene.terrain));
                }
            }
        }

        // Build GPU draw calls for every non-empty chunk.
        let mut draws = Vec::new();
        for cy in y_range {
            for cz in z_range.clone() {
                for cx in x_range.clone() {
                    let chunk_pos = IVec3::new(cx, cy, cz);
                    let chunk = world.get_chunk(&chunk_pos).unwrap();
                    let (verts, idx) = build_chunk_mesh(chunk, chunk_pos, &world);
                    if verts.is_empty() { continue; }

                    let vertex_buf = gpu.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label:    Some("bench vbuf"),
                            contents: bytemuck::cast_slice(&verts),
                            usage:    wgpu::BufferUsages::VERTEX,
                        }
                    );
                    let index_buf = gpu.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label:    Some("bench ibuf"),
                            contents: bytemuck::cast_slice(&idx),
                            usage:    wgpu::BufferUsages::INDEX,
                        }
                    );
                    let origin = [
                        (cx * CHUNK_SIZE_I) as f32,
                        (cy * CHUNK_SIZE_I) as f32,
                        (cz * CHUNK_SIZE_I) as f32,
                        0.0f32,
                    ];
                    let chunk_buf = gpu.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label:    Some("bench chunk uniform"),
                            contents: bytemuck::bytes_of(&ChunkUniform { origin }),
                            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        }
                    );
                    let chunk_bind_group = gpu.device.create_bind_group(
                        &wgpu::BindGroupDescriptor {
                            label:   Some("bench chunk bg"),
                            layout:  &self.pipeline.chunk_bind_group_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding:  0,
                                resource: chunk_buf.as_entire_binding(),
                            }],
                        }
                    );
                    draws.push(ChunkDrawCall {
                        vertex_count: verts.len() as u64,
                        index_count:  idx.len() as u32,
                        vertex_buf, index_buf, chunk_bind_group,
                    });
                }
            }
        }

        log::info!("setup_scene '{}': {} non-empty draw calls", scene.id, draws.len());

        let (_rt, render_view) = make_render_target(&gpu.device);
        let (_dt, depth_view)  = make_depth_target(&gpu.device);

        NaiveScene { draws, render_view, depth_view }
    }

    fn render_frame(
        &mut self,
        gpu:        &GpuContext,
        scene_data: &mut NaiveScene,
        _camera:    &Camera,
        camera_buf: &wgpu::Buffer,
        collector:  &mut MetricsCollector,
    ) {
        // Rebuild camera bind group pointing at the shared camera_buf each frame.
        let camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("bench naive camera bg frame"),
            layout:  &self.camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("bench naive frame") }
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bench naive pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scene_data.render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.53, g: 0.81, b: 0.98, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &scene_data.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes:    None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &camera_bg, &[]);

            for draw in &scene_data.draws {
                pass.set_bind_group(1, &draw.chunk_bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buf.slice(..));
                pass.set_index_buffer(draw.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..draw.index_count, 0, 0..1);
                collector.record_draw(draw.vertex_count, draw.index_count as u64);
            }
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn run_benchmarks() {
    log::info!("=== Naive renderer benchmark ===");

    let gpu = match GpuContext::new_headless(wgpu::Features::empty()) {
        Ok(g)  => g,
        Err(GpuError::NoAdapter) => {
            eprintln!("No GPU adapter — cannot benchmark.");
            std::process::exit(1);
        }
        Err(e) => { eprintln!("GPU error: {e}"); std::process::exit(1); }
    };
    log::info!("GPU: {}", gpu.adapter_info());

    let camera_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("bench camera"),
        size:               CameraUniform::SIZE,
        usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut renderer = NaiveBenchRenderer::new(&gpu);
    run_all_scenes(&mut renderer, &gpu, &camera_buf);
}