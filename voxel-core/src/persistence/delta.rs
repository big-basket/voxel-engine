use rkyv::{Archive, Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub struct VoxelDelta {
    pub index: u16,
    pub value: u8,
}

#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct ChunkDelta {
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    pub deltas: Vec<VoxelDelta>,
}

impl ChunkDelta {
    pub fn new(chunk_x: i32, chunk_y: i32, chunk_z: i32) -> Self {
        ChunkDelta {
            chunk_x,
            chunk_y,
            chunk_z,
            deltas: Vec::new(),
        }
    }

    pub fn push(&mut self, index: u16, value: u8) {
        self.deltas.push(VoxelDelta { index, value });
    }

    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }


    pub fn to_bytes(&self) -> Result<Vec<u8>, DeltaError> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map(|b| b.to_vec())
            .map_err(|e| DeltaError::Serialise(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<ChunkDelta, DeltaError> {
        let archived = rkyv::access::<ArchivedChunkDelta, rkyv::rancor::Error>(bytes)
            .map_err(|e| DeltaError::Validate(e.to_string()))?;
        rkyv::deserialize::<ChunkDelta, rkyv::rancor::Error>(archived)
            .map_err(|e| DeltaError::Deserialise(e.to_string()))
    }


    pub fn apply(&self, voxels: &mut [u8]) {
        for delta in &self.deltas {
            let idx = delta.index as usize;
            debug_assert!(idx < voxels.len(), "delta index {idx} out of range");
            if idx < voxels.len() {
                voxels[idx] = delta.value;
            }
        }
    }
}

pub fn diff_chunks(
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    base: &[u8],
    modified: &[u8],
) -> Option<ChunkDelta> {
    debug_assert_eq!(base.len(), modified.len());
    let mut delta = ChunkDelta::new(chunk_x, chunk_y, chunk_z);
    for (i, (&b, &m)) in base.iter().zip(modified.iter()).enumerate() {
        if b != m {
            delta.push(i as u16, m);
        }
    }
    if delta.is_empty() { None } else { Some(delta) }
}


#[derive(Debug)]
pub enum DeltaError {
    Serialise(String),
    Deserialise(String),
    Validate(String),
}

impl std::fmt::Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeltaError::Serialise(s) => write!(f, "serialise error: {s}"),
            DeltaError::Deserialise(s) => write!(f, "deserialise error: {s}"),
            DeltaError::Validate(s) => write!(f, "validate error: {s}"),
        }
    }
}

impl std::error::Error for DeltaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_delta_roundtrip() {
        let delta = ChunkDelta::new(0, 0, 0);
        assert!(delta.is_empty());
        let bytes = delta.to_bytes().expect("serialise");
        let loaded = ChunkDelta::from_bytes(&bytes).expect("deserialise");
        assert!(loaded.is_empty());
        assert_eq!(loaded.chunk_x, 0);
    }

    #[test]
    fn delta_roundtrip_with_edits() {
        let mut delta = ChunkDelta::new(3, -1, 7);
        delta.push(0, 1);
        delta.push(100, 5);
        delta.push(32767, 255);

        let bytes = delta.to_bytes().expect("serialise");
        let loaded = ChunkDelta::from_bytes(&bytes).expect("deserialise");

        assert_eq!(loaded.chunk_x, 3);
        assert_eq!(loaded.chunk_y, -1);
        assert_eq!(loaded.chunk_z, 7);
        assert_eq!(loaded.deltas.len(), 3);
        assert_eq!(loaded.deltas[0], VoxelDelta { index: 0,     value: 1   });
        assert_eq!(loaded.deltas[1], VoxelDelta { index: 100,   value: 5   });
        assert_eq!(loaded.deltas[2], VoxelDelta { index: 32767, value: 255 });
    }

    #[test]
    fn apply_delta_modifies_correct_indices() {
        let mut voxels = vec![0u8; 32_768];
        let mut delta = ChunkDelta::new(0, 0, 0);
        delta.push(10, 1);   // stone
        delta.push(500, 3);  // grass

        delta.apply(&mut voxels);
        assert_eq!(voxels[10], 1);
        assert_eq!(voxels[500], 3);
        assert_eq!(voxels[0], 0);
    }

    #[test]
    fn diff_chunks_finds_differences() {
        let base = vec![0u8; 32_768];
        let mut modified = base.clone();
        modified[42] = 1;
        modified[1000] = 3;

        let delta = diff_chunks(0, 0, 0, &base, &modified).expect("should have delta");
        assert_eq!(delta.deltas.len(), 2);

        let mut reconstructed = base.clone();
        delta.apply(&mut reconstructed);
        assert_eq!(reconstructed, modified);
    }

    #[test]
    fn diff_identical_chunks_returns_none() {
        let base = vec![1u8; 32_768];
        let result = diff_chunks(0, 0, 0, &base, &base.clone());
        assert!(result.is_none());
    }

    #[test]
    fn diff_and_apply_roundtrip() {
        let base = {
            let mut b = vec![0u8; 32_768];
            for i in 0..(32 * 32 * 8) {
                b[i] = 1;
            }
            b
        };

        let mut modified = base.clone();
        modified[0] = 0;
        modified[5] = 0;
        modified[32 * 32 * 9] = 2;

        let delta = diff_chunks(0, 0, 0, &base, &modified).unwrap();
        assert_eq!(delta.deltas.len(), 3);

        let bytes = delta.to_bytes().unwrap();
        let reloaded = ChunkDelta::from_bytes(&bytes).unwrap();

        let mut reconstructed = base.clone();
        reloaded.apply(&mut reconstructed);
        assert_eq!(reconstructed, modified);
    }
}