use std::collections::HashMap;
use glam::IVec3;

use super::chunk::{Chunk, CHUNK_SIZE_I};
use super::voxel::VoxelId;

/// Chunk positions are in chunk-space 
/// A chunk at position (0, 0, 0) covers world voxels (0..32, 0..32, 0..32).
/// Negative chunk positions are supported
#[derive(Default)]
pub struct World {
    pub chunks: HashMap<IVec3, Chunk>,
}

impl World {
    pub fn new() -> Self {
        World::default()
    }


    pub fn insert_chunk(&mut self, pos: IVec3, chunk: Chunk) {
        self.chunks.insert(pos, chunk);
    }

    pub fn remove_chunk(&mut self, pos: &IVec3) -> Option<Chunk> {
        self.chunks.remove(pos)
    }

    pub fn get_chunk(&self, pos: &IVec3) -> Option<&Chunk> {
        self.chunks.get(pos)
    }

    pub fn get_chunk_mut(&mut self, pos: &IVec3) -> Option<&mut Chunk> {
        self.chunks.get_mut(pos)
    }

    pub fn get_or_insert_chunk(&mut self, pos: IVec3) -> &mut Chunk {
        self.chunks.entry(pos).or_insert_with(Chunk::empty)
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn dirty_chunks(&self) -> Vec<IVec3> {
        self.chunks
            .iter()
            .filter(|(_, c)| c.dirty)
            .map(|(pos, _)| *pos)
            .collect()
    }

    pub fn get_voxel(&self, world_pos: IVec3) -> VoxelId {
        let (chunk_pos, local) = world_to_chunk(world_pos);
        self.chunks
            .get(&chunk_pos)
            .map(|c| c.get(local.x as usize, local.y as usize, local.z as usize))
            .unwrap_or(VoxelId::AIR)
    }

    pub fn set_voxel(&mut self, world_pos: IVec3, id: VoxelId) -> bool {
        let (chunk_pos, local) = world_to_chunk(world_pos);
        if let Some(chunk) = self.chunks.get_mut(&chunk_pos) {
            chunk.set(local.x as usize, local.y as usize, local.z as usize, id);
            true
        } else {
            false
        }
    }

    pub fn set_voxel_or_create(&mut self, world_pos: IVec3, id: VoxelId) {
        let (chunk_pos, local) = world_to_chunk(world_pos);
        let chunk = self.chunks.entry(chunk_pos).or_insert_with(Chunk::empty);
        chunk.set(local.x as usize, local.y as usize, local.z as usize, id);
    }
}


pub fn world_to_chunk(world_pos: IVec3) -> (IVec3, IVec3) {
    let chunk_pos = world_pos.div_euclid(IVec3::splat(CHUNK_SIZE_I));
    let local_pos = world_pos.rem_euclid(IVec3::splat(CHUNK_SIZE_I));
    (chunk_pos, local_pos)
}

pub fn chunk_to_world(chunk_pos: IVec3, local: IVec3) -> IVec3 {
    chunk_pos * CHUNK_SIZE_I + local
}

pub fn chunk_pos_of(world_pos: IVec3) -> IVec3 {
    world_pos.div_euclid(IVec3::splat(CHUNK_SIZE_I))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::chunk::CHUNK_SIZE;

    #[test]
    fn world_to_chunk_positive() {
        let (cp, lp) = world_to_chunk(IVec3::new(33, 0, 0));
        assert_eq!(cp, IVec3::new(1, 0, 0));
        assert_eq!(lp, IVec3::new(1, 0, 0));
    }

    #[test]
    fn world_to_chunk_origin() {
        let (cp, lp) = world_to_chunk(IVec3::ZERO);
        assert_eq!(cp, IVec3::ZERO);
        assert_eq!(lp, IVec3::ZERO);
    }

    #[test]
    fn world_to_chunk_negative() {
        let (cp, lp) = world_to_chunk(IVec3::new(-1, 0, 0));
        assert_eq!(cp, IVec3::new(-1, 0, 0));
        assert_eq!(lp, IVec3::new(31, 0, 0));
    }

    #[test]
    fn world_to_chunk_negative_boundary() {
        let (cp, lp) = world_to_chunk(IVec3::new(-32, 0, 0));
        assert_eq!(cp, IVec3::new(-1, 0, 0));
        assert_eq!(lp, IVec3::new(0, 0, 0));
    }

    #[test]
    fn chunk_to_world_roundtrip() {
        for chunk_x in -2..=2 {
            for local_x in 0..CHUNK_SIZE_I {
                let chunk_pos = IVec3::new(chunk_x, 0, 0);
                let local = IVec3::new(local_x, 0, 0);
                let world = chunk_to_world(chunk_pos, local);
                let (cp2, lp2) = world_to_chunk(world);
                assert_eq!(cp2, chunk_pos, "chunk_x={chunk_x} local_x={local_x}");
                assert_eq!(lp2, local,     "chunk_x={chunk_x} local_x={local_x}");
            }
        }
    }

    #[test]
    fn world_get_set_voxel() {
        let mut world = World::new();
        let pos = IVec3::new(5, 10, 15);

        assert_eq!(world.get_voxel(pos), VoxelId::AIR);

        assert!(!world.set_voxel(pos, VoxelId::STONE));

        world.insert_chunk(IVec3::ZERO, Chunk::empty());
        assert!(world.set_voxel(pos, VoxelId::STONE));
        assert_eq!(world.get_voxel(pos), VoxelId::STONE);
    }

    #[test]
    fn set_voxel_or_create_inserts_chunk() {
        let mut world = World::new();
        world.set_voxel_or_create(IVec3::new(100, 0, 0), VoxelId::DIRT);
        assert_eq!(world.get_voxel(IVec3::new(100, 0, 0)), VoxelId::DIRT);
    }

    #[test]
    fn dirty_chunks_tracking() {
        let mut world = World::new();
        world.insert_chunk(IVec3::new(0, 0, 0), Chunk::empty());
        world.insert_chunk(IVec3::new(1, 0, 0), Chunk::empty());

        assert_eq!(world.dirty_chunks().len(), 0);

        world.set_voxel(IVec3::new(0, 0, 0), VoxelId::STONE);
        let dirty = world.dirty_chunks();
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0], IVec3::new(0, 0, 0));
    }

    #[test]
    fn get_or_insert_creates_empty() {
        let mut world = World::new();
        let chunk = world.get_or_insert_chunk(IVec3::new(5, 0, 5));
        assert!(chunk.is_empty());
        assert_eq!(world.chunk_count(), 1);
    }

    #[test]
    fn world_cross_chunk_boundary() {
        let mut world = World::new();
        let a = IVec3::new(31, 0, 0);
        let b = IVec3::new(32, 0, 0);

        world.set_voxel_or_create(a, VoxelId::STONE);
        world.set_voxel_or_create(b, VoxelId::DIRT);

        assert_eq!(world.get_voxel(a), VoxelId::STONE);
        assert_eq!(world.get_voxel(b), VoxelId::DIRT);
        assert_eq!(world.chunk_count(), 2);
    }
}