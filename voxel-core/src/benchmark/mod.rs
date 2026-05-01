pub mod metrics;
pub mod recorder;
pub mod runner;
pub mod scenes;

pub use metrics::{FrameMetrics, MetricsCollector, MetricsSummary};
pub use recorder::{Recorder, RecorderError};
pub use runner::{
    BenchRenderer, run_all_scenes,
    make_render_target, make_depth_target,
    BENCH_WIDTH, BENCH_HEIGHT, RENDER_FORMAT, DEPTH_FORMAT,
};
pub use scenes::{BenchmarkScene, SceneKind};