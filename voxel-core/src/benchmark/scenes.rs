use glam::Vec3;
use crate::gen::TerrainParams;

/// The three benchmark scenarios from the project proposal.
/// Each scene is self-contained and reproducible from its parameters alone.
#[derive(Debug, Clone)]
pub struct BenchmarkScene {
    /// Short identifier used in CSV output filenames.
    pub id: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// Terrain generation parameters — seed makes scenes reproducible.
    pub terrain: TerrainParams,
    /// Camera starting position for this scene.
    pub camera_pos: Vec3,
    /// Camera look direction (normalised).
    pub camera_forward: Vec3,
    /// How many frames to measure before recording results.
    /// First N frames are discarded as warmup.
    pub warmup_frames: u32,
    /// How many frames to actually measure.
    pub measure_frames: u32,
    /// Scene-specific parameters.
    pub kind: SceneKind,
}

#[derive(Debug, Clone)]
pub enum SceneKind {
    /// Scene 1: static view of a high-density area at long draw distance.
    /// Camera is fixed — measures peak throughput for a worst-case view.
    StaticHighDensity {
        /// Number of chunks to load in each X/Z direction from origin.
        draw_radius: i32,
        /// Number of chunk layers to load vertically.
        vertical_layers: i32,
    },

    /// Scene 2: dynamic updates — voxels are modified every frame and
    /// the mesh must be rebuilt. Measures remesh latency under edit load.
    DynamicRemesh {
        /// Number of voxels modified per frame.
        edits_per_frame: u32,
        /// World-space region within which edits are applied.
        edit_radius: f32,
    },

    /// Scene 3: progressive stress test — voxel count increases each frame
    /// until FPS drops below 30. Records the maximum sustainable voxel count.
    StressTest {
        /// Number of additional voxels added per frame.
        voxels_per_step: u32,
        /// FPS floor — test ends when average drops below this.
        fps_floor: f32,
    },
}

/// Returns the three canonical benchmark scenes defined in the project proposal.
pub fn all_scenes() -> [BenchmarkScene; 3] {
    [
        BenchmarkScene {
            id: "static_high_density",
            description: "Static view — high-density scene, long draw distance, peak throughput",
            terrain: TerrainParams {
                seed: 12345,
                sea_level: 32,
                amplitude: 28.0,
                frequency: 0.012,
                octaves: 5,
                persistence: 0.5,
                lacunarity: 2.0,
            },
            // Elevated position looking across a wide landscape
            camera_pos: Vec3::new(0.0, 120.0, -200.0),
            camera_forward: Vec3::new(0.0, -0.25, 1.0).normalize(),
            warmup_frames: 60,
            measure_frames: 300,
            kind: SceneKind::StaticHighDensity {
                // 10-chunk radius = 21×21 = 441 surface chunks
                // At ~32K voxels/chunk this is ~14M voxels in the draw set
                draw_radius: 10,
                vertical_layers: 4,
            },
        },

        BenchmarkScene {
            id: "dynamic_remesh",
            description: "Dynamic updates — continuous voxel edits, measures remesh latency",
            terrain: TerrainParams {
                seed: 99999,
                sea_level: 32,
                amplitude: 16.0,
                frequency: 0.015,
                octaves: 4,
                persistence: 0.5,
                lacunarity: 2.0,
            },
            camera_pos: Vec3::new(0.0, 80.0, -60.0),
            camera_forward: Vec3::new(0.0, -0.25, 1.0).normalize(),
            warmup_frames: 30,
            measure_frames: 300,
            kind: SceneKind::DynamicRemesh {
                // 512 edits/frame forces frequent remeshing across many chunks
                edits_per_frame: 512,
                edit_radius: 48.0,
            },
        },

        BenchmarkScene {
            id: "stress_test",
            description: "Stress test — increasing voxel count until FPS drops below 30",
            terrain: TerrainParams {
                seed: 42,
                sea_level: 32,
                amplitude: 8.0,
                frequency: 0.02,
                octaves: 3,
                persistence: 0.5,
                lacunarity: 2.0,
            },
            camera_pos: Vec3::new(0.0, 120.0, -150.0),
            camera_forward: Vec3::new(0.0, -0.3, 1.0).normalize(),
            warmup_frames: 30,
            // Enough frames to reach the 30fps floor even on fast hardware
            measure_frames: 1200,
            kind: SceneKind::StressTest {
                // Each step adds one full chunk ring — coarser steps = faster ramp
                voxels_per_step: 4096,
                fps_floor: 30.0,
            },
        },
    ]
}