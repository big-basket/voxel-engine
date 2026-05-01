/// Unified benchmark runner — shared frame loop, scene iteration, and result
/// writing for both the naive and optimised renderers.
///
/// # Architecture
///
/// voxel-core cannot depend on either renderer crate (that would be a
/// circular dependency). Instead, callers implement `BenchRenderer` and pass
/// it into `run_all_scenes`. The trait is defined here and contains only the
/// wgpu operations that are identical between both renderers — uploading the
/// camera uniform, building a command encoder, dispatching work, and polling.
///
/// Renderer-specific setup (pipeline creation, mesh upload, indirect buffers)
/// lives in `renderer-naive/src/bench.rs` and
/// `renderer-optimised/src/bench.rs` respectively. Those files are now thin
/// shims: they implement `BenchRenderer`, call `run_all_scenes`, and handle
/// nothing else.

use crate::{
    benchmark::{BenchmarkScene, MetricsCollector, Recorder},
    benchmark::scenes::all_scenes,
    camera::{Camera, CameraUniform},
    gpu::{GpuContext, write_uniform},
};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A renderer that can be benchmarked headlessly.
///
/// Implementations provide:
/// - One-time scene setup (`setup_scene`) — uploads geometry, allocates GPU
///   buffers, returns an opaque handle the frame loop passes back.
/// - Per-frame work (`render_frame`) — dispatches GPU commands and records
///   the metrics for that frame.
pub trait BenchRenderer {
    /// Opaque per-scene state (mesh buffers, pipeline bind groups, etc.).
    type Scene: Send;

    /// Name shown in log output and written to result filenames.
    fn name(&self) -> &str;

    /// Directory results are written to.
    fn results_dir(&self) -> &str;

    /// Builds GPU resources for one benchmark scene.
    /// Called once per scene, before the frame loop.
    fn setup_scene(
        &mut self,
        gpu:   &GpuContext,
        scene: &BenchmarkScene,
    ) -> Self::Scene;

    /// Renders one frame into the off-screen target.
    /// `collector` is live — call `record_draw` inside with the geometry counts.
    /// The frame timing is handled by `run_all_scenes`; do not call
    /// `begin_frame` / `end_frame` here.
    fn render_frame(
        &mut self,
        gpu:        &GpuContext,
        scene_data: &mut Self::Scene,
        camera:     &Camera,
        camera_buf: &wgpu::Buffer,
        collector:  &mut MetricsCollector,
    );
}

// ── Shared loop ───────────────────────────────────────────────────────────────

pub const BENCH_WIDTH:  u32 = 1280;
pub const BENCH_HEIGHT: u32 = 720;

/// Runs all three canonical benchmark scenes through `renderer`, writes
/// per-frame CSVs and summary JSONs to `renderer.results_dir()`.
pub fn run_all_scenes<R: BenchRenderer>(
    renderer: &mut R,
    gpu:      &GpuContext,
    camera_buf: &wgpu::Buffer,
) {
    let recorder = Recorder::new(renderer.name(), renderer.results_dir());
    let scenes   = all_scenes();

    for scene in &scenes {
        log::info!(
            "=== {} | Scene: {} ===",
            renderer.name(), scene.id
        );

        let mut camera = Camera::new(BENCH_WIDTH as f32 / BENCH_HEIGHT as f32);
        camera.position = scene.camera_pos;
        camera.forward  = scene.camera_forward;

        let mut scene_data  = renderer.setup_scene(gpu, scene);
        let total_frames    = scene.warmup_frames + scene.measure_frames;
        let mut collector   = MetricsCollector::new();

        for frame_idx in 0..total_frames {
            let measuring = frame_idx >= scene.warmup_frames;

            // Upload camera uniform before every frame so the GPU sees the
            // correct view-projection matrix.
            write_uniform(
                &gpu.queue,
                camera_buf,
                &CameraUniform::from_camera(&camera),
            );

            if measuring { collector.begin_frame(); }

            renderer.render_frame(
                gpu,
                &mut scene_data,
                &camera,
                camera_buf,
                &mut collector,
            );

            // Block until the GPU has finished so frame time is wall-clock
            // accurate. Both renderers use this same synchronisation point.
            gpu.device.poll(wgpu::Maintain::Wait);

            if measuring { collector.end_frame(0); }
        }

        let summary = collector.summarise();
        log::info!(
            "  avg_fps={:.1}  1%_low={:.1}  avg_ms={:.2}  \
             triangles={}  draws={}",
            summary.avg_fps,
            summary.one_pct_low_fps,
            summary.avg_frame_ms,
            summary.avg_triangle_count,
            summary.avg_draw_calls,
        );

        if let Err(e) = recorder.write_all(
            &scene.id,
            &scene.description,
            collector.frames(),
            &summary,
        ) {
            log::error!(
                "Failed to write results for {}: {e}", &scene.id
            );
        }
    }

    log::info!(
        "Benchmark complete. Results in '{}'",
        renderer.results_dir()
    );
}

// ── Shared GPU target helpers ─────────────────────────────────────────────────

pub const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub const DEPTH_FORMAT:  wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub fn make_render_target(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("bench render target"),
        size:            wgpu::Extent3d {
            width: BENCH_WIDTH, height: BENCH_HEIGHT, depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          RENDER_FORMAT,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats:    &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

pub fn make_depth_target(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("bench depth"),
        size:            wgpu::Extent3d {
            width: BENCH_WIDTH, height: BENCH_HEIGHT, depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          DEPTH_FORMAT,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats:    &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}