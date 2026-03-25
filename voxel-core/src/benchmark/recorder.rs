use std::fs;
use std::path::{Path, PathBuf};

use super::metrics::{FrameMetrics, MetricsSummary};

/// Writes benchmark results to the `results/` directory.
///
/// Per-run output:
///   results/{renderer}_{scene_id}_frames.csv  — one row per frame
///   results/{renderer}_{scene_id}_summary.json — aggregated stats
pub struct Recorder {
    output_dir: PathBuf,
    renderer_name: String,
}

impl Recorder {
    /// Creates a new recorder.
    /// `renderer_name` should be `"naive"` or `"optimised"` — used in filenames.
    /// `output_dir` is created if it does not exist.
    pub fn new(renderer_name: impl Into<String>, output_dir: impl AsRef<Path>) -> Self {
        let output_dir = output_dir.as_ref().to_path_buf();
        fs::create_dir_all(&output_dir).expect("create results directory");
        Recorder {
            output_dir,
            renderer_name: renderer_name.into(),
        }
    }

    /// Writes the per-frame CSV for a scene.
    pub fn write_frames(
        &self,
        scene_id: &str,
        frames: &[FrameMetrics],
    ) -> Result<PathBuf, RecorderError> {
        let path = self
            .output_dir
            .join(format!("{}_{}_frames.csv", self.renderer_name, scene_id));

        let mut out = String::from(
            "frame,frame_time_ms,fps,triangle_count,vertex_count,draw_calls,vram_bytes\n",
        );

        for (i, f) in frames.iter().enumerate() {
            out.push_str(&format!(
                "{},{:.4},{:.4},{},{},{},{}\n",
                i,
                f.frame_time_ms,
                f.fps,
                f.triangle_count,
                f.vertex_count,
                f.draw_calls,
                f.vram_bytes,
            ));
        }

        fs::write(&path, out).map_err(RecorderError::Io)?;
        log::info!("Wrote {} frame rows to {}", frames.len(), path.display());
        Ok(path)
    }

    /// Writes the summary JSON for a scene.
    pub fn write_summary(
        &self,
        scene_id: &str,
        scene_description: &str,
        summary: &MetricsSummary,
    ) -> Result<PathBuf, RecorderError> {
        let path = self
            .output_dir
            .join(format!("{}_{}_summary.json", self.renderer_name, scene_id));

        let json = format!(
            r#"{{
  "renderer": "{renderer}",
  "scene_id": "{scene_id}",
  "description": "{description}",
  "frame_count": {frame_count},
  "avg_fps": {avg_fps:.4},
  "one_pct_low_fps": {one_pct_low:.4},
  "avg_frame_ms": {avg_frame_ms:.4},
  "avg_triangle_count": {avg_triangles},
  "avg_draw_calls": {avg_draws},
  "peak_vram_mb": {peak_vram_mb:.2}
}}
"#,
            renderer = self.renderer_name,
            scene_id = scene_id,
            description = scene_description,
            frame_count = summary.frame_count,
            avg_fps = summary.avg_fps,
            one_pct_low = summary.one_pct_low_fps,
            avg_frame_ms = summary.avg_frame_ms,
            avg_triangles = summary.avg_triangle_count,
            avg_draws = summary.avg_draw_calls,
            peak_vram_mb = summary.peak_vram_mb(),
        );

        fs::write(&path, json).map_err(RecorderError::Io)?;
        log::info!("Wrote summary to {}", path.display());
        Ok(path)
    }

    /// Convenience: writes both frames CSV and summary JSON in one call.
    pub fn write_all(
        &self,
        scene_id: &str,
        scene_description: &str,
        frames: &[FrameMetrics],
        summary: &MetricsSummary,
    ) -> Result<(), RecorderError> {
        self.write_frames(scene_id, frames)?;
        self.write_summary(scene_id, scene_description, summary)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum RecorderError {
    Io(std::io::Error),
}

impl std::fmt::Display for RecorderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecorderError::Io(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for RecorderError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::metrics::{FrameMetrics, MetricsSummary};

    fn dummy_frames(n: usize) -> Vec<FrameMetrics> {
        (0..n)
            .map(|_i| FrameMetrics {
                frame_time_ms: 16.666,
                fps: 60.0,
                vram_bytes: 1024 * 1024 * 200,
                triangle_count: 1_000_000,
                vertex_count: 2_000_000,
                draw_calls: 50,
            })
            .collect()
    }

    fn dummy_summary() -> MetricsSummary {
        MetricsSummary {
            frame_count: 300,
            avg_fps: 60.0,
            one_pct_low_fps: 45.0,
            avg_frame_ms: 16.666,
            avg_triangle_count: 1_000_000,
            avg_draw_calls: 50,
            peak_vram_bytes: 1024 * 1024 * 200,
        }
    }

    #[test]
    fn write_frames_creates_csv() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rec = Recorder::new("naive", dir.path());
        let frames = dummy_frames(5);
        let path = rec.write_frames("static_high_density", &frames).expect("write");

        assert!(path.exists());
        let content = fs::read_to_string(&path).expect("read");
        assert!(content.starts_with("frame,frame_time_ms"));
        assert_eq!(content.lines().count(), 6); // header + 5 data rows
    }

    #[test]
    fn write_summary_creates_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rec = Recorder::new("naive", dir.path());
        let summary = dummy_summary();
        let path = rec
            .write_summary("static_high_density", "test scene", &summary)
            .expect("write");

        assert!(path.exists());
        let content = fs::read_to_string(&path).expect("read");
        assert!(content.contains("\"renderer\": \"naive\""));
        assert!(content.contains("\"avg_fps\": 60.0000"));
        assert!(content.contains("\"one_pct_low_fps\": 45.0000"));
    }

    #[test]
    fn write_all_creates_both_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rec = Recorder::new("optimised", dir.path());
        let frames = dummy_frames(10);
        let summary = dummy_summary();
        rec.write_all("stress_test", "stress", &frames, &summary)
            .expect("write_all");

        assert!(dir.path().join("optimised_stress_test_frames.csv").exists());
        assert!(dir.path().join("optimised_stress_test_summary.json").exists());
    }

    #[test]
    fn filename_uses_renderer_and_scene() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rec = Recorder::new("naive", dir.path());
        let path = rec
            .write_frames("dynamic_remesh", &dummy_frames(1))
            .expect("write");
        assert!(path.file_name().unwrap().to_str().unwrap()
            .starts_with("naive_dynamic_remesh"));
    }
}