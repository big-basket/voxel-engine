use std::time::{Duration, Instant};

/// All metrics recorded for a single frame.
#[derive(Debug, Clone)]
pub struct FrameMetrics {
    /// Wall-clock frame time in milliseconds.
    pub frame_time_ms: f64,
    /// Instantaneous FPS derived from frame_time_ms.
    pub fps: f64,
    /// GPU VRAM used in bytes (queried from wgpu if available).
    pub vram_bytes: u64,
    /// Total triangle count submitted to the GPU this frame.
    pub triangle_count: u64,
    /// Total vertex count submitted.
    pub vertex_count: u64,
    /// Total draw call count.
    pub draw_calls: u32,
}

/// Accumulates per-frame metrics across a benchmark run and computes summary stats.
#[derive(Debug, Default)]
pub struct MetricsCollector {
    frames: Vec<FrameMetrics>,
    frame_start: Option<Instant>,
    /// Running triangle count for the current frame (reset each frame).
    pub current_triangles: u64,
    pub current_vertices: u64,
    pub current_draw_calls: u32,
}

impl MetricsCollector {
    pub fn new() -> Self {
        MetricsCollector::default()
    }

    /// Call at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.frame_start = Some(Instant::now());
        self.current_triangles = 0;
        self.current_vertices = 0;
        self.current_draw_calls = 0;
    }

    /// Register geometry being submitted this frame.
    pub fn record_draw(&mut self, vertex_count: u64, index_count: u64) {
        self.current_vertices += vertex_count;
        self.current_triangles += index_count / 3;
        self.current_draw_calls += 1;
    }

    /// Call at the end of each frame to commit the frame's metrics.
    /// `vram_bytes` should come from `device.global_report()` if available,
    /// otherwise pass 0.
    pub fn end_frame(&mut self, vram_bytes: u64) {
        let elapsed = self
            .frame_start
            .take()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO);

        let frame_time_ms = elapsed.as_secs_f64() * 1000.0;
        let fps = if frame_time_ms > 0.0 { 1000.0 / frame_time_ms } else { 0.0 };

        self.frames.push(FrameMetrics {
            frame_time_ms,
            fps,
            vram_bytes,
            triangle_count: self.current_triangles,
            vertex_count: self.current_vertices,
            draw_calls: self.current_draw_calls,
        });
    }

    /// Number of frames recorded so far.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns all recorded frames.
    pub fn frames(&self) -> &[FrameMetrics] {
        &self.frames
    }

    /// Computes summary statistics over all recorded frames.
    pub fn summarise(&self) -> MetricsSummary {
        if self.frames.is_empty() {
            return MetricsSummary::default();
        }

        let n = self.frames.len() as f64;

        let avg_fps = self.frames.iter().map(|f| f.fps).sum::<f64>() / n;
        let avg_frame_ms = self.frames.iter().map(|f| f.frame_time_ms).sum::<f64>() / n;

        // 1% low: average of the worst 1% of frame times (slowest = highest ms)
        let mut frame_times: Vec<f64> = self.frames.iter().map(|f| f.frame_time_ms).collect();
        frame_times.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
        let one_pct_count = ((self.frames.len() as f64 * 0.01).ceil() as usize).max(1);
        let one_pct_low_fps = {
            let avg_slow_ms = frame_times[..one_pct_count].iter().sum::<f64>()
                / one_pct_count as f64;
            if avg_slow_ms > 0.0 { 1000.0 / avg_slow_ms } else { 0.0 }
        };

        let avg_triangles = self.frames.iter().map(|f| f.triangle_count).sum::<u64>()
            / self.frames.len() as u64;
        let avg_draw_calls = (self.frames.iter().map(|f| f.draw_calls as u64).sum::<u64>()
            / self.frames.len() as u64) as u32;
        let peak_vram = self.frames.iter().map(|f| f.vram_bytes).max().unwrap_or(0);

        MetricsSummary {
            frame_count: self.frames.len(),
            avg_fps,
            one_pct_low_fps,
            avg_frame_ms,
            avg_triangle_count: avg_triangles,
            avg_draw_calls,
            peak_vram_bytes: peak_vram,
        }
    }
}

/// Summary statistics for a complete benchmark run.
#[derive(Debug, Clone, Default)]
pub struct MetricsSummary {
    pub frame_count: usize,
    pub avg_fps: f64,
    /// Average FPS of the worst 1% of frames.
    pub one_pct_low_fps: f64,
    pub avg_frame_ms: f64,
    pub avg_triangle_count: u64,
    pub avg_draw_calls: u32,
    pub peak_vram_bytes: u64,
}

impl MetricsSummary {
    pub fn peak_vram_mb(&self) -> f64 {
        self.peak_vram_bytes as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_collector(frame_times_ms: &[f64]) -> MetricsCollector {
        let mut c = MetricsCollector::new();
        for &ms in frame_times_ms {
            c.frames.push(FrameMetrics {
                frame_time_ms: ms,
                fps: 1000.0 / ms,
                vram_bytes: 0,
                triangle_count: 1000,
                vertex_count: 2000,
                draw_calls: 10,
            });
        }
        c
    }

    #[test]
    fn avg_fps_is_correct() {
        // 16.67ms = 60fps, 33.33ms = 30fps → avg ≈ 45fps
        let c = make_collector(&[16.666, 33.333]);
        let s = c.summarise();
        assert!((s.avg_fps - 45.0).abs() < 1.0, "avg_fps={}", s.avg_fps);
    }

    #[test]
    fn one_pct_low_picks_worst_frames() {
        // 100 frames: 99 at 16ms (60fps), 1 at 100ms (10fps)
        let mut times: Vec<f64> = vec![16.666; 99];
        times.push(100.0);
        let c = make_collector(&times);
        let s = c.summarise();
        // 1% low should be close to 10fps (one slow frame)
        assert!(s.one_pct_low_fps < 15.0, "1% low={}", s.one_pct_low_fps);
        assert!(s.avg_fps > 55.0, "avg_fps={}", s.avg_fps);
    }

    #[test]
    fn empty_collector_returns_default() {
        let c = MetricsCollector::new();
        let s = c.summarise();
        assert_eq!(s.frame_count, 0);
        assert_eq!(s.avg_fps, 0.0);
    }

    #[test]
    fn record_draw_accumulates() {
        let mut c = MetricsCollector::new();
        c.begin_frame();
        c.record_draw(4, 6);   // 1 quad = 2 triangles
        c.record_draw(8, 12);  // 2 quads = 4 triangles
        assert_eq!(c.current_triangles, 6); // (6+12)/3
        assert_eq!(c.current_vertices, 12);
        assert_eq!(c.current_draw_calls, 2);
    }

    #[test]
    fn begin_frame_resets_counters() {
        let mut c = MetricsCollector::new();
        c.begin_frame();
        c.record_draw(100, 150);
        c.end_frame(0);
        c.begin_frame();
        assert_eq!(c.current_triangles, 0);
        assert_eq!(c.current_draw_calls, 0);
    }
}