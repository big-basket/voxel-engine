use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::gen::TerrainParams;

/// The camera position and orientation for a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub position: [f32; 3],
    pub forward:  [f32; 3],
}

/// Scene-specific parameters — tagged union so serde can round-trip it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SceneKind {
    StaticHighDensity {
        draw_radius:    i32,
        vertical_layers: i32,
    },
    DynamicRemesh {
        edits_per_frame: u32,
        edit_radius:     f32,
    },
    StressTest {
        voxels_per_step: u32,
        fps_floor:       f32,
    },
}

/// One benchmark scenario — fully serialisable so it can be loaded from JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScene {
    /// Short identifier used in CSV/JSON output filenames.
    pub id:          String,
    /// Human-readable description written into the summary JSON.
    pub description: String,
    /// Terrain generation parameters.
    pub terrain:     TerrainParams,
    /// Camera config.
    pub camera:      CameraConfig,
    /// Frames rendered but not measured at the start of each scene.
    pub warmup_frames:  u32,
    /// Frames actually measured.
    pub measure_frames: u32,
    /// Scene type and its specific parameters.
    pub kind: SceneKind,
}

impl BenchmarkScene {
    /// Convenience accessor — returns camera position as Vec3.
    pub fn camera_pos(&self) -> Vec3 {
        Vec3::from(self.camera.position)
    }

    /// Convenience accessor — returns normalised camera forward as Vec3.
    pub fn camera_forward(&self) -> Vec3 {
        Vec3::from(self.camera.forward).normalize()
    }
}

/// Full benchmark configuration — the top-level JSON structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub scenes: Vec<BenchmarkScene>,
}

impl BenchmarkConfig {
    /// Loads config from `benchmark_config.json` at `config_path`.
    /// Returns the hardcoded defaults if the file is not found.
    pub fn load_or_default(config_path: &std::path::Path) -> Self {
        if config_path.exists() {
            let content = std::fs::read_to_string(config_path)
                .expect("read benchmark_config.json");
            match serde_json::from_str(&content) {
                Ok(cfg) => {
                    log::info!("Loaded benchmark config from {}", config_path.display());
                    return cfg;
                }
                Err(e) => {
                    log::warn!(
                        "Failed to parse {}: {e}. Using defaults.",
                        config_path.display()
                    );
                }
            }
        } else {
            log::info!(
                "No benchmark_config.json found at {} — using defaults.",
                config_path.display()
            );
        }
        Self::default_config()
    }

    /// Writes the current config to disk as pretty-printed JSON.
    /// Useful for generating the initial file to edit from.
    pub fn write_default(path: &std::path::Path) -> std::io::Result<()> {
        let cfg = Self::default_config();
        let json = serde_json::to_string_pretty(&cfg).expect("serialise config");
        std::fs::write(path, json)
    }

    fn default_config() -> Self {
        BenchmarkConfig {
            scenes: vec![
                BenchmarkScene {
                    id: "static_high_density".into(),
                    description: "Static view — high-density scene, long draw distance, peak throughput".into(),
                    terrain: TerrainParams {
                        seed: 12345,
                        sea_level: 32,
                        amplitude: 28.0,
                        frequency: 0.012,
                        octaves: 5,
                        persistence: 0.5,
                        lacunarity: 2.0,
                    },
                    camera: CameraConfig {
                        position: [0.0, 120.0, -200.0],
                        forward:  [0.0, -0.25, 1.0],
                    },
                    warmup_frames:  60,
                    measure_frames: 300,
                    kind: SceneKind::StaticHighDensity {
                        draw_radius:     10,
                        vertical_layers: 4,
                    },
                },
                BenchmarkScene {
                    id: "dynamic_remesh".into(),
                    description: "Dynamic updates — continuous voxel edits, measures remesh latency".into(),
                    terrain: TerrainParams {
                        seed: 99999,
                        sea_level: 32,
                        amplitude: 16.0,
                        frequency: 0.015,
                        octaves: 4,
                        persistence: 0.5,
                        lacunarity: 2.0,
                    },
                    camera: CameraConfig {
                        position: [0.0, 80.0, -60.0],
                        forward:  [0.0, -0.25, 1.0],
                    },
                    warmup_frames:  30,
                    measure_frames: 300,
                    kind: SceneKind::DynamicRemesh {
                        edits_per_frame: 512,
                        edit_radius:     48.0,
                    },
                },
                BenchmarkScene {
                    id: "stress_test".into(),
                    description: "Stress test — increasing voxel count until FPS drops below 30".into(),
                    terrain: TerrainParams {
                        seed: 42,
                        sea_level: 32,
                        amplitude: 8.0,
                        frequency: 0.02,
                        octaves: 3,
                        persistence: 0.5,
                        lacunarity: 2.0,
                    },
                    camera: CameraConfig {
                        position: [0.0, 120.0, -150.0],
                        forward:  [0.0, -0.3, 1.0],
                    },
                    warmup_frames:  30,
                    measure_frames: 1200,
                    kind: SceneKind::StressTest {
                        voxels_per_step: 4096,
                        fps_floor:       30.0,
                    },
                },
            ],
        }
    }
}

// Keep TerrainParams serialisable — add derives there via the gen module.
// The impl is here so callers only need to import scenes::{BenchmarkConfig, BenchmarkScene}.