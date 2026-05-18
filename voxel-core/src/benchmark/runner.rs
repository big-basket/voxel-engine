/// Unified benchmark runner 

use crate::{
    benchmark::{BenchmarkScene, MetricsCollector, Recorder},
    benchmark::scenes::BenchmarkConfig,
    camera::{Camera, CameraUniform},
    gpu::{GpuContext, write_uniform},
};


pub trait BenchRenderer {
    type Scene: Send;

    fn name(&self) -> &str;

    fn results_dir(&self) -> &str;

    fn setup_scene(
        &mut self,
        gpu:   &GpuContext,
        scene: &BenchmarkScene,
    ) -> Self::Scene;

    fn render_frame(
        &mut self,
        gpu:        &GpuContext,
        scene_data: &mut Self::Scene,
        camera:     &Camera,
        camera_buf: &wgpu::Buffer,
        collector:  &mut MetricsCollector,
    );
}


pub const BENCH_WIDTH:  u32 = 1280;
pub const BENCH_HEIGHT: u32 = 720;


pub fn run_all_scenes<R: BenchRenderer>(
    renderer: &mut R,
    gpu:      &GpuContext,
    camera_buf: &wgpu::Buffer,
) {
    let recorder = Recorder::new(renderer.name(), renderer.results_dir());
    let config   = BenchmarkConfig::load_or_default(
        std::path::Path::new("benchmark_config.json")
    );

    for scene in &config.scenes {
        log::info!(
            "=== {} | Scene: {} ===",
            renderer.name(), scene.id
        );

        let mut camera = Camera::new(BENCH_WIDTH as f32 / BENCH_HEIGHT as f32);
        camera.position = scene.camera_pos();
        camera.forward  = scene.camera_forward();

        let mut scene_data  = renderer.setup_scene(gpu, scene);
        let total_frames    = scene.warmup_frames + scene.measure_frames;
        let mut collector   = MetricsCollector::new();

        for frame_idx in 0..total_frames {
            let measuring = frame_idx >= scene.warmup_frames;


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