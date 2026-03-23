use glam::{IVec3, Vec3};

use crate::world::{VoxelId, World};

/// Result of a brush raycast.
#[derive(Debug, Clone, PartialEq)]
pub struct RayHit {
    /// The world-space voxel position that was hit.
    pub voxel_pos: IVec3,
    /// The world-space position of the voxel just before the hit.
    /// Used by `place` to know where to put a new block.
    pub prev_pos: IVec3,
    /// Distance from the ray origin to the hit.
    pub distance: f32,
}

/// Casts a ray from `origin` in `direction` (normalised) and returns the
/// first solid voxel hit within `max_distance` world units, if any.
///
/// Uses a DDA (Digital Differential Analyser) voxel traversal — it steps
/// through voxels exactly, with no floating-point intersection errors.
pub fn raycast(
    world: &World,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
) -> Option<RayHit> {
    // Current voxel position (integer)
    let mut pos = IVec3::new(
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );

    let step = IVec3::new(
        if direction.x >= 0.0 { 1 } else { -1 },
        if direction.y >= 0.0 { 1 } else { -1 },
        if direction.z >= 0.0 { 1 } else { -1 },
    );

    // t_delta: how far along the ray to cross one full voxel in each axis.
    let t_delta = Vec3::new(
        if direction.x.abs() > 1e-9 { 1.0 / direction.x.abs() } else { f32::INFINITY },
        if direction.y.abs() > 1e-9 { 1.0 / direction.y.abs() } else { f32::INFINITY },
        if direction.z.abs() > 1e-9 { 1.0 / direction.z.abs() } else { f32::INFINITY },
    );

    // t_max: t-value at the next voxel boundary in each axis.
    let t_max = Vec3::new(
        if direction.x >= 0.0 {
            (pos.x as f32 + 1.0 - origin.x) / direction.x.abs().max(1e-9)
        } else {
            (origin.x - pos.x as f32) / direction.x.abs().max(1e-9)
        },
        if direction.y >= 0.0 {
            (pos.y as f32 + 1.0 - origin.y) / direction.y.abs().max(1e-9)
        } else {
            (origin.y - pos.y as f32) / direction.y.abs().max(1e-9)
        },
        if direction.z >= 0.0 {
            (pos.z as f32 + 1.0 - origin.z) / direction.z.abs().max(1e-9)
        } else {
            (origin.z - pos.z as f32) / direction.z.abs().max(1e-9)
        },
    );

    let mut t_max = t_max;
    let mut prev_pos = pos;
    let mut t = 0.0f32;

    loop {
        if t > max_distance {
            return None;
        }

        if world.get_voxel(pos).is_solid() {
            return Some(RayHit { voxel_pos: pos, prev_pos, distance: t });
        }

        prev_pos = pos;

        // Step along the axis with the smallest t_max.
        if t_max.x < t_max.y && t_max.x < t_max.z {
            t = t_max.x;
            t_max.x += t_delta.x;
            pos.x += step.x;
        } else if t_max.y < t_max.z {
            t = t_max.y;
            t_max.y += t_delta.y;
            pos.y += step.y;
        } else {
            t = t_max.z;
            t_max.z += t_delta.z;
            pos.z += step.z;
        }
    }
}

/// Removes the voxel at `hit.voxel_pos` (sets it to AIR).
/// Returns true if the chunk was loaded and the write succeeded.
pub fn remove(world: &mut World, hit: &RayHit) -> bool {
    world.set_voxel(hit.voxel_pos, VoxelId::AIR)
}

/// Places `id` at `hit.prev_pos` (the face in front of the hit voxel).
/// Returns false if the target chunk is not loaded.
pub fn place(world: &mut World, hit: &RayHit, id: VoxelId) -> bool {
    // Don't place on top of air (shouldn't happen from a real hit, but guard it).
    if id.is_air() {
        return false;
    }
    world.set_voxel(hit.prev_pos, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Chunk, VoxelId, World};

    /// Builds a world with a flat stone floor at y=0 in the origin chunk.
    fn flat_world() -> World {
        let mut world = World::new();
        let mut chunk = Chunk::empty();
        chunk.fill_layer(0, VoxelId::STONE);
        world.insert_chunk(IVec3::ZERO, chunk);
        world
    }

    #[test]
    fn ray_hits_floor_from_above() {
        let world = flat_world();
        // Cast straight down from (0.5, 5.0, 0.5)
        let hit = raycast(&world, Vec3::new(0.5, 5.0, 0.5), Vec3::NEG_Y, 20.0);
        assert!(hit.is_some(), "expected a hit on the floor");
        let hit = hit.unwrap();
        assert_eq!(hit.voxel_pos.y, 0, "should hit y=0 stone layer");
    }

    #[test]
    fn ray_misses_when_pointing_away() {
        let world = flat_world();
        // Cast upward — no ceiling, should miss
        let hit = raycast(&world, Vec3::new(0.5, 5.0, 0.5), Vec3::Y, 20.0);
        assert!(hit.is_none(), "upward ray should miss the floor");
    }

    #[test]
    fn ray_misses_beyond_max_distance() {
        let world = flat_world();
        // Floor is at y=0, origin at y=5, max_distance=2 — can't reach
        let hit = raycast(&world, Vec3::new(0.5, 5.0, 0.5), Vec3::NEG_Y, 2.0);
        assert!(hit.is_none(), "ray should not reach floor within max_distance=2");
    }

    #[test]
    fn prev_pos_is_above_hit_voxel() {
        let world = flat_world();
        let hit = raycast(&world, Vec3::new(0.5, 5.0, 0.5), Vec3::NEG_Y, 20.0)
            .expect("should hit");
        // The voxel in front of a floor hit (coming from above) should be y=1
        assert_eq!(hit.prev_pos.y, 1, "prev_pos should be y=1 above the floor");
    }

    #[test]
    fn remove_voxel_sets_air() {
        let mut world = flat_world();
        let hit = raycast(&world, Vec3::new(0.5, 5.0, 0.5), Vec3::NEG_Y, 20.0)
            .expect("should hit");
        let success = remove(&mut world, &hit);
        assert!(success);
        assert_eq!(world.get_voxel(hit.voxel_pos), VoxelId::AIR);
    }

    #[test]
    fn place_voxel_on_floor() {
        let mut world = flat_world();
        let hit = raycast(&world, Vec3::new(0.5, 5.0, 0.5), Vec3::NEG_Y, 20.0)
            .expect("should hit");
        let place_pos = hit.prev_pos;
        let success = place(&mut world, &hit, VoxelId::DIRT);
        assert!(success);
        assert_eq!(world.get_voxel(place_pos), VoxelId::DIRT);
    }

    #[test]
    fn place_air_is_rejected() {
        let mut world = flat_world();
        let hit = raycast(&world, Vec3::new(0.5, 5.0, 0.5), Vec3::NEG_Y, 20.0)
            .expect("should hit");
        let success = place(&mut world, &hit, VoxelId::AIR);
        assert!(!success, "placing AIR should be rejected");
    }

    #[test]
    fn ray_from_origin_inside_voxel_still_works() {
        // Start inside a solid voxel — the DDA should detect it immediately.
        let world = flat_world();
        let hit = raycast(&world, Vec3::new(0.5, 0.5, 0.5), Vec3::Y, 20.0);
        // The starting voxel (0,0,0) is stone, so it should be a hit at t≈0
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().voxel_pos, IVec3::new(0, 0, 0));
    }

    #[test]
    fn horizontal_ray_hits_wall() {
        let mut world = World::new();
        let mut chunk = Chunk::empty();
        // Place a wall at x=10 (all y, z=0 column)
        for y in 0..32usize {
            chunk.set(10, y, 0, VoxelId::STONE);
        }
        world.insert_chunk(IVec3::ZERO, chunk);

        let hit = raycast(
            &world,
            Vec3::new(0.5, 1.5, 0.5),
            Vec3::X,
            50.0,
        );
        assert!(hit.is_some(), "horizontal ray should hit wall at x=10");
        assert_eq!(hit.unwrap().voxel_pos.x, 10);
    }
}