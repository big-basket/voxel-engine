use super::voxel::VoxelId;

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I: i32 = CHUNK_SIZE as i32;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

#[derive(Clone)]
pub struct Chunk {
    voxels: Box<[u8; CHUNK_VOLUME]>,
    pub dirty: bool,
}

impl Chunk {
    pub fn empty() -> Self {
        Chunk {
            voxels: Box::new([0u8; CHUNK_VOLUME]),
            dirty: false,
        }
    }

    pub fn from_raw(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != CHUNK_VOLUME {
            return None;
        }
        let mut arr = [0u8; CHUNK_VOLUME];
        arr.copy_from_slice(bytes);
        Some(Chunk {
            voxels: Box::new(arr),
            dirty: false,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.voxels.as_ref()
    }


    #[inline]
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < CHUNK_SIZE, "x={x} out of range");
        debug_assert!(y < CHUNK_SIZE, "y={y} out of range");
        debug_assert!(z < CHUNK_SIZE, "z={z} out of range");
        x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
    }

    #[inline]
    pub fn coords(index: usize) -> (usize, usize, usize) {
        let x = index % CHUNK_SIZE;
        let z = (index / CHUNK_SIZE) % CHUNK_SIZE;
        let y = index / (CHUNK_SIZE * CHUNK_SIZE);
        (x, y, z)
    }


    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> VoxelId {
        if x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return VoxelId::AIR;
        }
        VoxelId(self.voxels[Self::index(x, y, z)])
    }

    #[inline]
    pub fn get_idx(&self, idx: usize) -> VoxelId {
        VoxelId(self.voxels[idx])
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, id: VoxelId) {
        if x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return;
        }
        let idx = Self::index(x, y, z);
        self.voxels[idx] = id.0;
        self.dirty = true;
    }

    pub fn is_empty(&self) -> bool {
        self.voxels.iter().all(|&v| v == 0)
    }

    pub fn solid_count(&self) -> usize {
        self.voxels.iter().filter(|&&v| v != 0).count()
    }

    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    pub fn fill(&mut self, id: VoxelId) {
        self.voxels.iter_mut().for_each(|v| *v = id.0);
        self.dirty = true;
    }

    pub fn fill_layer(&mut self, y: usize, id: VoxelId) {
        if y >= CHUNK_SIZE {
            return;
        }
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                self.voxels[Self::index(x, y, z)] = id.0;
            }
        }
        self.dirty = true;
    }
}

impl std::fmt::Debug for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Chunk {{ solid: {}, dirty: {} }}",
            self.solid_count(),
            self.dirty
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chunk_is_all_air() {
        let c = Chunk::empty();
        for i in 0..CHUNK_VOLUME {
            assert_eq!(c.get_idx(i), VoxelId::AIR);
        }
        assert!(c.is_empty());
    }

    #[test]
    fn set_and_get_roundtrip() {
        let mut c = Chunk::empty();
        c.set(1, 2, 3, VoxelId::STONE);
        assert_eq!(c.get(1, 2, 3), VoxelId::STONE);
        assert_eq!(c.get(0, 0, 0), VoxelId::AIR);
        assert!(c.dirty);
    }

    #[test]
    fn index_coords_roundtrip() {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let idx = Chunk::index(x, y, z);
                    let (rx, ry, rz) = Chunk::coords(idx);
                    assert_eq!((x, y, z), (rx, ry, rz));
                }
            }
        }
    }

    #[test]
    fn chunk_volume_is_correct() {
        assert_eq!(CHUNK_VOLUME, 32 * 32 * 32);
        assert_eq!(CHUNK_VOLUME, 32_768);
    }

    #[test]
    fn out_of_bounds_get_returns_air() {
        let c = Chunk::empty();
        // These should not panic and should return AIR
        assert_eq!(c.get(32, 0, 0), VoxelId::AIR);
        assert_eq!(c.get(0, 32, 0), VoxelId::AIR);
        assert_eq!(c.get(0, 0, 32), VoxelId::AIR);
    }

    #[test]
    fn out_of_bounds_set_is_ignored() {
        let mut c = Chunk::empty();
        c.set(32, 0, 0, VoxelId::STONE); // should not panic
        assert!(c.is_empty()); // should be unchanged
    }

    #[test]
    fn fill_layer_correct() {
        let mut c = Chunk::empty();
        c.fill_layer(0, VoxelId::DIRT);
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                assert_eq!(c.get(x, 0, z), VoxelId::DIRT, "x={x} z={z}");
                assert_eq!(c.get(x, 1, z), VoxelId::AIR,  "x={x} z={z}");
            }
        }
    }

    #[test]
    fn solid_count() {
        let mut c = Chunk::empty();
        assert_eq!(c.solid_count(), 0);
        c.fill_layer(0, VoxelId::STONE);
        assert_eq!(c.solid_count(), CHUNK_SIZE * CHUNK_SIZE);
    }

    #[test]
    fn from_raw_roundtrip() {
        let mut original = Chunk::empty();
        original.set(5, 10, 15, VoxelId::GRASS);
        original.set(0, 0, 0, VoxelId::STONE);

        let bytes = original.as_bytes().to_vec();
        let loaded = Chunk::from_raw(&bytes).expect("from_raw failed");

        assert_eq!(loaded.get(5, 10, 15), VoxelId::GRASS);
        assert_eq!(loaded.get(0, 0, 0), VoxelId::STONE);
        assert_eq!(loaded.get(1, 1, 1), VoxelId::AIR);
    }

    #[test]
    fn from_raw_rejects_wrong_size() {
        assert!(Chunk::from_raw(&[0u8; 100]).is_none());
        assert!(Chunk::from_raw(&[]).is_none());
    }

    #[test]
    fn mark_clean_clears_dirty() {
        let mut c = Chunk::empty();
        c.set(0, 0, 0, VoxelId::STONE);
        assert!(c.dirty);
        c.mark_clean();
        assert!(!c.dirty);
    }
}