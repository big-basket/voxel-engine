use bytemuck::{Pod, Zeroable};
use glam::IVec3;

use voxel_core::world::{Chunk, CHUNK_SIZE, CHUNK_SIZE_I, World};
#[cfg(test)]
use voxel_core::world::VoxelId;

// ── Vertex layout ─────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub packed_pos: u32,
    pub packed_data: u32,
}

impl Vertex {
    pub fn new(x: u8, y: u8, z: u8, face: u8, voxel_id: u8) -> Self {
        // 6 bits per axis: x[0..6] | y[6..12] | z[12..18]
        // 6 bits handles 0..63, which covers 0..32 (face corners go one past chunk size)
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
    // POS_X: viewed from +X, CCW = bottom-left, bottom-right, top-right, top-left
    [[1,0,1],[1,0,0],[1,1,0],[1,1,1]],
    // NEG_X: viewed from -X, CCW
    [[0,0,0],[0,0,1],[0,1,1],[0,1,0]],
    // POS_Y: viewed from above (+Y), CCW
    [[0,1,1],[1,1,1],[1,1,0],[0,1,0]],
    // NEG_Y: viewed from below (-Y), CCW
    [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
    // POS_Z: viewed from +Z, CCW
    [[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
    // NEG_Z: viewed from -Z, CCW
    [[1,0,0],[0,0,0],[0,1,0],[1,1,0]],
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

/// Builds a mesh for `chunk` at `chunk_pos`, using `world` to query neighbours
/// across chunk boundaries so no spurious inter-chunk faces are generated.
pub fn build_chunk_mesh(
    chunk: &Chunk,
    chunk_pos: IVec3,
    world: &World,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // World-space origin of this chunk's (0,0,0) voxel
    let origin = chunk_pos * CHUNK_SIZE_I;

    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let voxel = chunk.get(x, y, z);
                if voxel.is_air() {
                    continue;
                }

                for face in 0..6usize {
                    let [nx, ny, nz] = FACE_NEIGHBOURS[face];
                    let lbx = x as i32 + nx;
                    let lby = y as i32 + ny;
                    let lbz = z as i32 + nz;

                    let neighbour_solid = if lbx >= 0 && lbx < CHUNK_SIZE_I
                        && lby >= 0 && lby < CHUNK_SIZE_I
                        && lbz >= 0 && lbz < CHUNK_SIZE_I
                    {
                        // Neighbour is inside this chunk
                        chunk.get(lbx as usize, lby as usize, lbz as usize).is_solid()
                    } else {
                        // Neighbour is in an adjacent chunk — query world-space
                        let world_nb = origin + IVec3::new(lbx, lby, lbz);
                        world.get_voxel(world_nb).is_solid()
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
    use voxel_core::world::{Chunk, World};

    fn empty_world() -> World { World::new() }

    #[test]
    fn empty_chunk_produces_no_mesh() {
        let mut world = World::new();
        let chunk = Chunk::empty();
        world.insert_chunk(IVec3::ZERO, Chunk::empty());
        let (verts, idx) = build_chunk_mesh(&chunk, IVec3::ZERO, &world);
        assert!(verts.is_empty());
        assert!(idx.is_empty());
    }

    #[test]
    fn single_voxel_has_six_faces() {
        let mut world = World::new();
        let mut chunk = Chunk::empty();
        chunk.set(1, 1, 1, VoxelId::STONE);
        world.insert_chunk(IVec3::ZERO, chunk.clone());
        let (verts, idx) = build_chunk_mesh(&chunk, IVec3::ZERO, &world);
        assert_eq!(verts.len(), 24);
        assert_eq!(idx.len(), 36);
    }

    #[test]
    fn two_adjacent_voxels_hide_shared_face() {
        let mut world = World::new();
        let mut chunk = Chunk::empty();
        chunk.set(0, 0, 0, VoxelId::STONE);
        chunk.set(1, 0, 0, VoxelId::STONE);
        world.insert_chunk(IVec3::ZERO, chunk.clone());
        let (verts, _) = build_chunk_mesh(&chunk, IVec3::ZERO, &world);
        assert_eq!(verts.len(), 10 * 4);
    }

    #[test]
    fn full_layer_exposes_only_top_bottom_and_sides() {
        let mut world = World::new();
        let mut chunk = Chunk::empty();
        chunk.fill_layer(1, VoxelId::STONE);
        world.insert_chunk(IVec3::ZERO, chunk.clone());
        let (verts, _) = build_chunk_mesh(&chunk, IVec3::ZERO, &world);
        // top: 1024, bottom: 1024, sides: 128 = 2176 faces
        assert_eq!(verts.len(), 2176 * 4);
    }

    #[test]
    fn inter_chunk_face_is_hidden() {
        // Two adjacent chunks, both solid on the boundary — the shared face must be hidden
        let mut world = World::new();

        let mut chunk_a = Chunk::empty();
        chunk_a.fill_layer(0, VoxelId::STONE); // bottom layer solid
        world.insert_chunk(IVec3::new(0, 0, 0), chunk_a.clone());

        let mut chunk_b = Chunk::empty();
        chunk_b.fill_layer(31, VoxelId::STONE); // top layer solid
        world.insert_chunk(IVec3::new(0, -1, 0), chunk_b.clone());

        // chunk_b's top layer (y=31 local) is adjacent to chunk_a's bottom (y=0 local)
        // Neither should generate a face on that shared boundary
        let (verts_a, _) = build_chunk_mesh(&chunk_a, IVec3::new(0,0,0), &world);
        let (verts_b, _) = build_chunk_mesh(&chunk_b, IVec3::new(0,-1,0), &world);

        // Count NEG_Y faces on chunk_a bottom layer
        let neg_y_faces_a = verts_a.iter()
            .filter(|v| (v.packed_data & 0x7) == 3) // face 3 = NEG_Y
            .count() / 4;
        // Count POS_Y faces on chunk_b top layer
        let pos_y_faces_b = verts_b.iter()
            .filter(|v| (v.packed_data & 0x7) == 2) // face 2 = POS_Y
            .count() / 4;

        assert_eq!(neg_y_faces_a, 0, "chunk_a bottom should have no NEG_Y faces (neighbour is solid)");
        assert_eq!(pos_y_faces_b, 0, "chunk_b top should have no POS_Y faces (neighbour is solid)");
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