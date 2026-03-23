/// A voxel identifier.
///
/// `0` is always AIR — transparent, not rendered, not stored in deltas.
/// Values 1–255 map to entries in the block palette.
/// Using u8 keeps the chunk grid at 32 KiB (32³ = 32 768 bytes), which fits
/// comfortably in L2 cache on the RX 6800M and most desktop CPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct VoxelId(pub u8);

impl VoxelId {
    pub const AIR: VoxelId = VoxelId(0);
    pub const STONE: VoxelId = VoxelId(1);
    pub const DIRT: VoxelId = VoxelId(2);
    pub const GRASS: VoxelId = VoxelId(3);
    pub const SAND: VoxelId = VoxelId(4);
    pub const WATER: VoxelId = VoxelId(5);

    #[inline]
    pub fn is_air(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_solid(self) -> bool {
        self.0 != 0
    }
}

impl From<u8> for VoxelId {
    fn from(v: u8) -> Self {
        VoxelId(v)
    }
}

impl From<VoxelId> for u8 {
    fn from(v: VoxelId) -> u8 {
        v.0
    }
}

impl std::fmt::Display for VoxelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VoxelId({})", self.0)
    }
}

/// Bitmask of which faces of a voxel are exposed to air.
/// Used by the mesher to skip hidden faces without a second lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FaceMask(pub u8);

impl FaceMask {
    pub const NONE: FaceMask = FaceMask(0);
    pub const POS_X: FaceMask = FaceMask(1 << 0);
    pub const NEG_X: FaceMask = FaceMask(1 << 1);
    pub const POS_Y: FaceMask = FaceMask(1 << 2); // top
    pub const NEG_Y: FaceMask = FaceMask(1 << 3); // bottom
    pub const POS_Z: FaceMask = FaceMask(1 << 4);
    pub const NEG_Z: FaceMask = FaceMask(1 << 5);
    pub const ALL: FaceMask = FaceMask(0b0011_1111);

    #[inline]
    pub fn set(&mut self, face: FaceMask) {
        self.0 |= face.0;
    }

    #[inline]
    pub fn has(self, face: FaceMask) -> bool {
        self.0 & face.0 != 0
    }
}

// ── Face index constants (used for array indexing in the mesher) ───────────────
pub const FACE_POS_X: usize = 0;
pub const FACE_NEG_X: usize = 1;
pub const FACE_POS_Y: usize = 2;
pub const FACE_NEG_Y: usize = 3;
pub const FACE_POS_Z: usize = 4;
pub const FACE_NEG_Z: usize = 5;

pub const FACE_NORMALS: [[i32; 3]; 6] = [
    [ 1,  0,  0], // POS_X
    [-1,  0,  0], // NEG_X
    [ 0,  1,  0], // POS_Y
    [ 0, -1,  0], // NEG_Y
    [ 0,  0,  1], // POS_Z
    [ 0,  0, -1], // NEG_Z
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_is_zero() {
        assert_eq!(VoxelId::AIR.0, 0);
        assert!(VoxelId::AIR.is_air());
        assert!(!VoxelId::AIR.is_solid());
    }

    #[test]
    fn solid_voxels() {
        assert!(!VoxelId::STONE.is_air());
        assert!(VoxelId::STONE.is_solid());
    }

    #[test]
    fn roundtrip_u8() {
        for i in 0u8..=255 {
            assert_eq!(u8::from(VoxelId::from(i)), i);
        }
    }

    #[test]
    fn face_mask_set_and_has() {
        let mut m = FaceMask::NONE;
        assert!(!m.has(FaceMask::POS_X));
        m.set(FaceMask::POS_X);
        m.set(FaceMask::NEG_Y);
        assert!(m.has(FaceMask::POS_X));
        assert!(m.has(FaceMask::NEG_Y));
        assert!(!m.has(FaceMask::POS_Z));
    }

    #[test]
    fn face_mask_all_covers_six_faces() {
        // ALL should have exactly 6 bits set
        assert_eq!(FaceMask::ALL.0.count_ones(), 6);
    }
}