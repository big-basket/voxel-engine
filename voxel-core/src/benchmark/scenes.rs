use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::gen::TerrainParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub position: [f32; 3],
    pub forward:  [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SceneKind {
    StaticHighDensity {
        draw_radius:     i32,
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

impl SceneKind {

    pub fn spatial_extent(&self) -> (i32, i32) {
        match self {
            SceneKind::StaticHighDensity { draw_radius, vertical_layers } => {
                (*draw_radius, *vertical_layers)
            }
            _ => (8, 4),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScene {
    pub id:          String,
    pub description: String,
    pub terrain:     TerrainParams,
    pub camera:      CameraConfig,
    pub warmup_frames:  u32,
    pub measure_frames: u32,
    pub kind: SceneKind,
}

impl BenchmarkScene {
    pub fn camera_pos(&self) -> Vec3 {
        Vec3::from(self.camera.position)
    }

    pub fn camera_forward(&self) -> Vec3 {
        Vec3::from(self.camera.forward).normalize()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub scenes: Vec<BenchmarkScene>,
}

impl BenchmarkConfig {
    /// Loads config from `benchmark_config.json` at `config_path`.
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

