use bytemuck::{Pod, Zeroable};
use glam::IVec3;

use voxel_core::world::{Chunk, CHUNK_SIZE};
#[cfg(test)]
use voxel_core::world::VoxelId;

// ── Vertex layout ─────────────────────────────────────────────────────────────

/// A single vertex in the naive mesh.
/// packed_pos:  x[0..5] | y[5..10] | z[10..15]  (5 bits each, 0..32)
/// packed_data: face[0..3] | voxel_id[3..11]
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub packed_pos: u32,
    pub packed_data: u32,
}

impl Vertex {
    pub fn new(x: u8, y: u8, z: u8, face: u8, voxel_id: u8) -> Self {
        let packed_pos = (x as u32) | ((y as u32) << 6) | ((z as u32) << 12);
        let packed_data = (face as u32) | ((voxel_id as u32) << 3);
        Vertex { packed_pos, packed_data }
    }

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: 4,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

// ── Face definitions ──────────────────────────────────────────────────────────

const FACE_VERTICES: [[[i32; 3]; 4]; 6] = [
    // POS_X: (1,0,0) -> (1,0,1) -> (1,1,1) -> (1,1,0)
    [[1,0,0],[1,0,1],[1,1,1],[1,1,0]], 
    // NEG_X: (0,0,1) -> (0,0,0) -> (0,1,0) -> (0,1,1)
    [[0,0,1],[0,0,0],[0,1,0],[0,1,1]], 
    // POS_Y: (0,1,0) -> (1,1,0) -> (1,1,1) -> (0,1,1)
    [[0,1,0],[1,1,0],[1,1,1],[0,1,1]], 
    // NEG_Y: (0,0,0) -> (0,0,1) -> (1,0,1) -> (1,0,0)
    [[0,0,0],[0,0,1],[1,0,1],[1,0,0]], 
    // POS_Z: (1,0,1) -> (0,0,1) -> (0,1,1) -> (1,1,1)
    [[1,0,1],[0,0,1],[0,1,1],[1,1,1]], 
    // NEG_Z: (0,0,0) -> (1,0,0) -> (1,1,0) -> (0,1,0)
    [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], 
];

const FACE_NEIGHBOURS: [[i32; 3]; 6] = [
    [ 1, 0, 0],
    [-1, 0, 0],
    [ 0, 1, 0],
    [ 0,-1, 0],
    [ 0, 0, 1],
    [ 0, 0,-1],
];

const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

// ── Mesh builder ──────────────────────────────────────────────────────────────

pub fn build_chunk_mesh(chunk: &Chunk, _chunk_pos: IVec3) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let voxel = chunk.get(x, y, z);
                if voxel.is_air() {
                    continue;
                }

                for face in 0..6usize {
                    let [nx, ny, nz] = FACE_NEIGHBOURS[face];
                    let nbx = x as i32 + nx;
                    let nby = y as i32 + ny;
                    let nbz = z as i32 + nz;

                    let neighbour_solid = if nbx >= 0 && nbx < CHUNK_SIZE as i32
                        && nby >= 0 && nby < CHUNK_SIZE as i32
                        && nbz >= 0 && nbz < CHUNK_SIZE as i32
                    {
                        chunk.get(nbx as usize, nby as usize, nbz as usize).is_solid()
                    } else {
                        false
                    };

                    if neighbour_solid {
                        continue;
                    }

                    let base = vertices.len() as u32;
                    for &[cx, cy, cz] in &FACE_VERTICES[face] {
                        vertices.push(Vertex::new(
                            (x as i32 + cx) as u8,
                            (y as i32 + cy) as u8,
                            (z as i32 + cz) as u8,
                            face as u8,
                            voxel.0,
                        ));
                    }
                    for &i in &QUAD_INDICES {
                        indices.push(base + i);
                    }
                }
            }
        }
    }

    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::world::Chunk;

    #[test]
    fn empty_chunk_produces_no_mesh() {
        let chunk = Chunk::empty();
        let (verts, idx) = build_chunk_mesh(&chunk, IVec3::ZERO);
        assert!(verts.is_empty());
        assert!(idx.is_empty());
    }

    #[test]
    fn single_voxel_has_six_faces() {
        let mut chunk = Chunk::empty();
        chunk.set(1, 1, 1, VoxelId::STONE);
        let (verts, idx) = build_chunk_mesh(&chunk, IVec3::ZERO);
        assert_eq!(verts.len(), 24);
        assert_eq!(idx.len(), 36);
    }

    #[test]
    fn two_adjacent_voxels_hide_shared_face() {
        let mut chunk = Chunk::empty();
        chunk.set(0, 0, 0, VoxelId::STONE);
        chunk.set(1, 0, 0, VoxelId::STONE);
        let (verts, _) = build_chunk_mesh(&chunk, IVec3::ZERO);
        assert_eq!(verts.len(), 10 * 4);
    }

    #[test]
    fn full_layer_exposes_only_top_bottom_and_sides() {
        let mut chunk = Chunk::empty();
        chunk.fill_layer(1, VoxelId::STONE);
        let (verts, _) = build_chunk_mesh(&chunk, IVec3::ZERO);
        // top: 1024, bottom: 1024, sides: 128 = 2176 faces
        assert_eq!(verts.len(), 2176 * 4);
    }

    #[test]
    fn vertex_packing_roundtrip() {
        let v = Vertex::new(31, 15, 7, 3, 255);
        let x = (v.packed_pos & 0x3F) as u8;
        let y = ((v.packed_pos >> 6) & 0x3F) as u8;
        let z = ((v.packed_pos >> 12) & 0x3F) as u8;
        let face = (v.packed_data & 0x7) as u8;
        let id = ((v.packed_data >> 3) & 0xFF) as u8;
        assert_eq!((x, y, z, face, id), (31, 15, 7, 3, 255));
    }
}