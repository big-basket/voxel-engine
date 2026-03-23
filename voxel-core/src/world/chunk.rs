use super::voxel::VoxelId;

/// Side length of a chunk in voxels. Must be a power of two.
pub const CHUNK_SIZE: usize = 32;
/// CHUNK_SIZE as i32 for signed arithmetic.
pub const CHUNK_SIZE_I: i32 = CHUNK_SIZE as i32;
/// Total number of voxels in a chunk (32³ = 32 768).
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// A chunk is a 32×32×32 block of voxels stored as a flat byte array.
///
/// Layout: `index = x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE`
/// (x is the inner-most axis for cache locality when iterating a face slice).
///
/// The array lives on the heap (Box) so a Chunk is pointer-sized on the stack.
/// At 32 KiB per chunk this keeps stack usage reasonable even when the world
/// module holds many chunks in a HashMap.
#[derive(Clone)]
pub struct Chunk {
    /// Raw voxel data. 0 = air, anything else = solid block.
    voxels: Box<[u8; CHUNK_VOLUME]>,
    /// True if this chunk has been modified since the last persistence flush.
    pub dirty: bool,
}

impl Chunk {
    /// Creates an empty (all-air) chunk.
    pub fn empty() -> Self {
        Chunk {
            voxels: Box::new([0u8; CHUNK_VOLUME]),
            dirty: false,
        }
    }

    /// Creates a chunk from raw bytes (e.g. loaded from redb).
    /// Returns `None` if the slice is not exactly `CHUNK_VOLUME` bytes.
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

    /// Returns a reference to the raw voxel byte slice.
    /// Useful for serialisation without an extra copy.
    pub fn as_bytes(&self) -> &[u8] {
        self.voxels.as_ref()
    }

    // ── Coordinate helpers ──────────────────────────────────────────────────

    /// Converts (x, y, z) local chunk coordinates to a flat array index.
    ///
    /// # Panics (debug only)
    /// Panics if any coordinate is >= CHUNK_SIZE.
    #[inline]
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < CHUNK_SIZE, "x={x} out of range");
        debug_assert!(y < CHUNK_SIZE, "y={y} out of range");
        debug_assert!(z < CHUNK_SIZE, "z={z} out of range");
        x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
    }

    /// Returns the (x, y, z) local coordinates for a flat index.
    #[inline]
    pub fn coords(index: usize) -> (usize, usize, usize) {
        let x = index % CHUNK_SIZE;
        let z = (index / CHUNK_SIZE) % CHUNK_SIZE;
        let y = index / (CHUNK_SIZE * CHUNK_SIZE);
        (x, y, z)
    }

    // ── Voxel accessors ─────────────────────────────────────────────────────

    /// Gets the voxel at local coordinates. Returns AIR for out-of-bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> VoxelId {
        if x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return VoxelId::AIR;
        }
        VoxelId(self.voxels[Self::index(x, y, z)])
    }

    /// Gets the voxel at a flat index.
    #[inline]
    pub fn get_idx(&self, idx: usize) -> VoxelId {
        VoxelId(self.voxels[idx])
    }

    /// Sets the voxel at local coordinates and marks the chunk dirty.
    /// Out-of-bounds writes are silently ignored.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, id: VoxelId) {
        if x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return;
        }
        let idx = Self::index(x, y, z);
        self.voxels[idx] = id.0;
        self.dirty = true;
    }

    /// Returns true if every voxel in this chunk is air.
    pub fn is_empty(&self) -> bool {
        self.voxels.iter().all(|&v| v == 0)
    }

    /// Returns the number of non-air voxels. Useful for stress-test metrics.
    pub fn solid_count(&self) -> usize {
        self.voxels.iter().filter(|&&v| v != 0).count()
    }

    /// Clears the dirty flag after a successful persistence flush.
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Fills the entire chunk with a single voxel type.
    pub fn fill(&mut self, id: VoxelId) {
        self.voxels.iter_mut().for_each(|v| *v = id.0);
        self.dirty = true;
    }

    /// Fills a y-slice (a horizontal layer) with a single voxel type.
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