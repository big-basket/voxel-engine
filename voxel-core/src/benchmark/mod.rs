pub mod metrics;
pub mod recorder;
pub mod scenes;

pub use metrics::{FrameMetrics, MetricsCollector, MetricsSummary};
pub use recorder::{Recorder, RecorderError};
pub use scenes::{BenchmarkConfig, BenchmarkScene, CameraConfig, SceneKind};