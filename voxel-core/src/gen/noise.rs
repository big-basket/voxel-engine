use noise::{NoiseFn, Perlin};
use glam::IVec3;

use crate::world::{Chunk, VoxelId, CHUNK_SIZE, CHUNK_SIZE_I};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainParams {
    pub seed: u32,
    pub sea_level: i32,
    pub amplitude: f64,
    pub frequency: f64,
    pub octaves: u32,
    pub persistence: f64,
    pub lacunarity: f64,
}

impl Default for TerrainParams {
    fn default() -> Self {
        TerrainParams {
            seed: 42,
            sea_level: 32,
            amplitude: 24.0,
            frequency: 0.015,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
        }
    }
}


pub fn generate_chunk(chunk_pos: IVec3, params: &TerrainParams) -> Chunk {
    let noise = Perlin::new(params.seed);
    let mut chunk = Chunk::empty();

    let world_origin_x = chunk_pos.x * CHUNK_SIZE_I;
    let world_origin_y = chunk_pos.y * CHUNK_SIZE_I;
    let world_origin_z = chunk_pos.z * CHUNK_SIZE_I;

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let world_x = world_origin_x + x as i32;
            let world_z = world_origin_z + z as i32;

            let surface_y = sample_height(world_x, world_z, params, &noise);

            for y in 0..CHUNK_SIZE {
                let world_y = world_origin_y + y as i32;

                let voxel = classify_voxel(world_y, surface_y, params.sea_level);
                if !voxel.is_air() {
                    chunk.set(x, y, z, voxel);
                }
            }
        }
    }

    chunk.mark_clean();
    chunk
}

pub fn sample_height(world_x: i32, world_z: i32, params: &TerrainParams, noise: &Perlin) -> i32 {
    let mut amplitude = params.amplitude;
    let mut frequency = params.frequency;
    let mut value = 0.0f64;
    let mut max_value = 0.0f64;

    for _ in 0..params.octaves {
        value += noise.get([world_x as f64 * frequency, world_z as f64 * frequency]) * amplitude;
        max_value += amplitude;
        amplitude *= params.persistence;
        frequency *= params.lacunarity;
    }

    let normalised = value / max_value; // in [-1, 1]
    let height = params.sea_level as f64 + normalised * params.amplitude;
    height.round() as i32
}

fn classify_voxel(world_y: i32, surface_y: i32, sea_level: i32) -> VoxelId {
    if world_y > surface_y {
        VoxelId::AIR
    } else if world_y == surface_y {
        VoxelId::GRASS
    } else if world_y >= surface_y - 3 {
        VoxelId::DIRT
    } else if world_y < sea_level - 20 {
        VoxelId::STONE
    } else {
        VoxelId::DIRT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noise::Perlin;

    fn default_noise() -> Perlin {
        Perlin::new(42)
    }

    #[test]
    fn chunk_is_deterministic() {
        let params = TerrainParams::default();
        let pos = IVec3::new(0, 0, 0);
        let a = generate_chunk(pos, &params);
        let b = generate_chunk(pos, &params);
        assert_eq!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn different_seeds_produce_different_terrain() {
        let params_a = TerrainParams { seed: 1, ..Default::default() };
        let params_b = TerrainParams { seed: 2, ..Default::default() };
        let pos = IVec3::new(5, 0, 5);
        let a = generate_chunk(pos, &params_a);
        let b = generate_chunk(pos, &params_b);
        assert_ne!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn generated_chunk_is_not_dirty() {
        let params = TerrainParams::default();
        let chunk = generate_chunk(IVec3::ZERO, &params);
        assert!(!chunk.dirty, "generated terrain should start clean");
    }

    #[test]
    fn surface_chunk_has_grass_on_top() {
        let params = TerrainParams { sea_level: 16, amplitude: 4.0, ..Default::default() };
        let chunk = generate_chunk(IVec3::new(0, 0, 0), &params);
        // At least one voxel somewhere in the chunk should be grass
        let has_grass = (0..CHUNK_SIZE).any(|y| {
            (0..CHUNK_SIZE).any(|z| {
                (0..CHUNK_SIZE).any(|x| chunk.get(x, y, z) == VoxelId::GRASS)
            })
        });
        assert!(has_grass, "surface chunk should contain some grass");
    }

    #[test]
    fn deep_chunk_is_all_stone() {
        let params = TerrainParams {
            sea_level: 32,
            amplitude: 4.0,
            ..Default::default()
        };
        let chunk = generate_chunk(IVec3::new(0, -10, 0), &params);
        assert!(!chunk.is_empty(), "deep chunk should be solid");
        let has_air = (0..CHUNK_SIZE).any(|y| {
            (0..CHUNK_SIZE).any(|z| {
                (0..CHUNK_SIZE).any(|x| chunk.get(x, y, z) == VoxelId::AIR)
            })
        });
        assert!(!has_air, "deep chunk should have no air");
    }

    #[test]
    fn sky_chunk_is_all_air() {
        let params = TerrainParams {
            sea_level: 32,
            amplitude: 4.0,
            ..Default::default()
        };
        let chunk = generate_chunk(IVec3::new(0, 10, 0), &params);
        assert!(chunk.is_empty(), "sky chunk should be all air");
    }

    #[test]
    fn height_is_near_sea_level() {
        let params = TerrainParams::default();
        let noise = default_noise();
        for x in -5..5 {
            for z in -5..5 {
                let h = sample_height(x * 16, z * 16, &params, &noise);
                let deviation = (h - params.sea_level).abs();
                assert!(
                    deviation <= params.amplitude as i32 + 2,
                    "height {h} too far from sea_level {} at ({x},{z})",
                    params.sea_level
                );
            }
        }
    }

    #[test]
    fn adjacent_chunks_share_no_seam() {
        let params = TerrainParams { sea_level: 16, amplitude: 4.0, ..Default::default() };
        let noise = Perlin::new(params.seed);

        let h_right = sample_height(31, 0, &params, &noise);
        let h_left_next = sample_height(32, 0, &params, &noise);

        assert!(h_right.abs() < 1000);
        assert!(h_left_next.abs() < 1000);
    }
}