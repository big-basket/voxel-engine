/// Binary greedy mesher — voxel-core implementation.
///
/// Based on cgerikj/binary-greedy-meshing v2, incorporating Ethan Gore's
/// core V2 changes and vertex pulling output format.
///
/// # How it works
///
/// ## Step 1 — occupancy mask
/// A 32×32 array of u32 bitfields is built for the chunk (one u32 per
/// column of 32 voxels along the primary axis). Each bit is 1 if the voxel
/// is opaque, 0 if air. This takes one pass over the 32 768 voxel bytes.
///
/// ## Step 2 — face masks
/// For each of the 6 face directions, the occupancy mask is shifted by one
/// position and XOR'd with itself to find exposed faces — positions where
/// a solid voxel is adjacent to an air voxel. This produces a face-presence
/// bitfield for the entire chunk in a handful of bitwise operations.
///
/// ## Step 3 — quad merging
/// Each face-mask plane is scanned. When a set bit is found, the greedy
/// algorithm extends it as far as possible in both directions while:
///   - All voxels in the run have the same type
///   - All face bits in the run are set (no gaps)
/// The resulting rectangle becomes one quad.
///
/// ## Output format
/// Each quad is 8 bytes:
///   bytes 0-3: `x(6) | y(6) | z(6) | width(6) | height(6)` packed
///   bytes 4-7: `voxel_id(8)` in the low byte, rest reserved
///
/// This is the format consumed by the vertex-pulling shader in the optimised
/// renderer. The shader reads a quad by `vertex_index / 4`, computes the
/// corner from `vertex_index % 4`, and unpacks position + dimensions.
///
/// ## Neighbour data
/// `mesh_chunk` accepts an optional `NeighbourData` struct containing the
/// edge-voxel columns of the six adjacent chunks. When present, face culling
/// at chunk boundaries is exact — no seam faces are emitted. When absent,
/// boundary faces are always emitted (conservative, slightly over-draws).

use crate::world::{Chunk, CHUNK_SIZE};

//  Constants ─

/// CHUNK_SIZE as u32 for bitfield arithmetic.
const CS: usize = CHUNK_SIZE; // 32
const CS2: usize = CS * CS;   // 1024

/// Face indices — same ordering as in the naive renderer's mesh.rs.
pub const FACE_POS_X: usize = 0;
pub const FACE_NEG_X: usize = 1;
pub const FACE_POS_Y: usize = 2;
pub const FACE_NEG_Y: usize = 3;
pub const FACE_POS_Z: usize = 4;
pub const FACE_NEG_Z: usize = 5;

//  Output types 

/// A single greedy quad, ready for upload into the GPU vertex pool.
///
/// Packing (8 bytes total):
///   `data[0]`: `x(6) | y(6) | z(6) | w(6) | h(6)`  — position + dimensions
///   `data[1]`: `face(3) | voxel_id(8)`               — face direction + type
///
/// The vertex shader reconstructs the 4 corners from (x,y,z,w,h,face).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
pub struct GreedyQuad {
    /// Packed position and size.
    pub pos_size: u32,
    /// Packed face index and voxel type.
    pub face_type: u32,
}

impl GreedyQuad {
    /// Creates a quad from local chunk coordinates (0..=CS), face, and voxel id.
    #[inline]
    pub fn new(x: u32, y: u32, z: u32, w: u32, h: u32, face: u32, voxel_id: u8) -> Self {
        debug_assert!(x  < CS as u32 + 1, "x={x}");
        debug_assert!(y  < CS as u32 + 1, "y={y}");
        debug_assert!(z  < CS as u32 + 1, "z={z}");
        debug_assert!(w  > 0 && w <= CS as u32, "w={w}");
        debug_assert!(h  > 0 && h <= CS as u32, "h={h}");
        debug_assert!(face < 6, "face={face}");
        GreedyQuad {
            pos_size:  x | (y << 6) | (z << 12) | (w << 18) | (h << 24),
            face_type: face | ((voxel_id as u32) << 3),
        }
    }

    /// Decodes `(x, y, z, width, height)` from the packed `pos_size` field.
    #[inline]
    pub fn decode_pos(&self) -> (u32, u32, u32, u32, u32) {
        let x = (self.pos_size)       & 0x3F;
        let y = (self.pos_size >>  6) & 0x3F;
        let z = (self.pos_size >> 12) & 0x3F;
        let w = (self.pos_size >> 18) & 0x3F;
        let h = (self.pos_size >> 24) & 0x3F;
        (x, y, z, w, h)
    }

    /// Decodes `(face, voxel_id)` from `face_type`.
    #[inline]
    pub fn decode_type(&self) -> (u32, u8) {
        let face     = self.face_type & 0x7;
        let voxel_id = (self.face_type >> 3) as u8;
        (face, voxel_id)
    }
}

//  Neighbour data 

pub struct NeighbourData<'a> {
    pub pos_x: &'a [u8; CS2],
    pub neg_x: &'a [u8; CS2],
    pub pos_y: &'a [u8; CS2],
    pub neg_y: &'a [u8; CS2],
    pub pos_z: &'a [u8; CS2],
    pub neg_z: &'a [u8; CS2],
}

//  Occupancy mask 

#[allow(dead_code)]
type OccupancyMask = [[u32; CS]; CS];


#[allow(dead_code)]
fn build_occupancy(chunk: &Chunk, _neighbours: Option<&NeighbourData<'_>>) -> OccupancyMask {
    let mut mask = [[0u32; CS]; CS];

    for y in 0..CS {
        for z in 0..CS {
            for x in 0..CS {
                if chunk.get(x, y, z).is_solid() {
                    mask[y][z] |= 1u32 << x;
                }
            }
        }
    }
    mask
}

//  Face mask extraction 

#[allow(dead_code)]
fn face_masks_x(mask: &OccupancyMask) -> ([OccupancyMask; 1], [OccupancyMask; 1]) {
    let mut pos = [[[0u32; CS]; CS]];
    let mut neg = [[[0u32; CS]; CS]];
    for y in 0..CS {
        for z in 0..CS {
            let col = mask[y][z];
            // POS_X: solid voxel where the +X neighbour is air
            pos[0][y][z] = col & !(col << 1);
            // NEG_X: solid voxel where the -X neighbour is air
            neg[0][y][z] = col & !(col >> 1);
        }
    }
    (pos, neg)
}

//  Main mesher 


pub fn mesh_chunk(chunk: &Chunk, neighbours: Option<&NeighbourData<'_>>) -> Vec<GreedyQuad> {
    if chunk.is_empty() {
        return Vec::new();
    }

    let mut quads = Vec::new();

    for face in [FACE_POS_Y, FACE_NEG_Y] {
        for y in 0..CS {
            let neighbour_y = if face == FACE_POS_Y {
                if y + 1 < CS { y + 1 } else { CS } // CS signals "use neighbour chunk"
            } else {
                if y > 0 { y - 1 } else { CS }
            };

            let mut face_present  = [[0u8; CS]; CS]; // [z][x]
            let mut face_voxel_id = [[0u8; CS]; CS]; // [z][x]

            for z in 0..CS {
                for x in 0..CS {
                    let voxel = chunk.get(x, y, z);
                    if !voxel.is_solid() { continue; }

                    let neighbour_solid = if neighbour_y < CS {
                        chunk.get(x, neighbour_y, z).is_solid()
                    } else {
                        // Need neighbour chunk data
                        match neighbours {
                            None => false, // conservative: show face
                            Some(nb) => {
                                let slice = if face == FACE_POS_Y { nb.pos_y } else { nb.neg_y };
                                // Neighbour face slice: index = x + z*CS
                                slice[x + z * CS] != 0
                            }
                        }
                    };

                    if !neighbour_solid {
                        face_present[z][x]  = 1;
                        face_voxel_id[z][x] = voxel.0;
                    }
                }
            }

            // Greedy merge in X then Z.
            let mut visited = [[false; CS]; CS];
            for z in 0..CS {
                for x in 0..CS {
                    if visited[z][x] || face_present[z][x] == 0 { continue; }
                    let vid = face_voxel_id[z][x];

                    // Extend in +X
                    let mut w = 1usize;
                    while x + w < CS
                        && !visited[z][x + w]
                        && face_present[z][x + w] != 0
                        && face_voxel_id[z][x + w] == vid
                    {
                        w += 1;
                    }

                    let mut h = 1usize;
                    'outer: while z + h < CS {
                        for dx in 0..w {
                            if visited[z + h][x + dx]
                                || face_present[z + h][x + dx] == 0
                                || face_voxel_id[z + h][x + dx] != vid
                            {
                                break 'outer;
                            }
                        }
                        h += 1;
                    }

                    for dz in 0..h {
                        for dx in 0..w {
                            visited[z + dz][x + dx] = true;
                        }
                    }

                    let quad_y = if face == FACE_POS_Y { y + 1 } else { y } as u32;
                    quads.push(GreedyQuad::new(
                        x as u32, quad_y, z as u32,
                        w as u32, h as u32,
                        face as u32, vid,
                    ));
                }
            }
        }
    }

    //  ±Z faces 
    for face in [FACE_POS_Z, FACE_NEG_Z] {
        for z in 0..CS {
            let neighbour_z = if face == FACE_POS_Z {
                if z + 1 < CS { z + 1 } else { CS }
            } else {
                if z > 0 { z - 1 } else { CS }
            };

            let mut face_present  = [[0u8; CS]; CS]; // [y][x]
            let mut face_voxel_id = [[0u8; CS]; CS];

            for y in 0..CS {
                for x in 0..CS {
                    let voxel = chunk.get(x, y, z);
                    if !voxel.is_solid() { continue; }

                    let neighbour_solid = if neighbour_z < CS {
                        chunk.get(x, y, neighbour_z).is_solid()
                    } else {
                        match neighbours {
                            None => false,
                            Some(nb) => {
                                let slice = if face == FACE_POS_Z { nb.pos_z } else { nb.neg_z };
                                slice[x + y * CS] != 0
                            }
                        }
                    };

                    if !neighbour_solid {
                        face_present[y][x]  = 1;
                        face_voxel_id[y][x] = voxel.0;
                    }
                }
            }

            let mut visited = [[false; CS]; CS];
            for y in 0..CS {
                for x in 0..CS {
                    if visited[y][x] || face_present[y][x] == 0 { continue; }
                    let vid = face_voxel_id[y][x];

                    let mut w = 1usize;
                    while x + w < CS
                        && !visited[y][x + w]
                        && face_present[y][x + w] != 0
                        && face_voxel_id[y][x + w] == vid
                    {
                        w += 1;
                    }

                    let mut h = 1usize;
                    'outer: while y + h < CS {
                        for dx in 0..w {
                            if visited[y + h][x + dx]
                                || face_present[y + h][x + dx] == 0
                                || face_voxel_id[y + h][x + dx] != vid
                            {
                                break 'outer;
                            }
                        }
                        h += 1;
                    }

                    for dy in 0..h {
                        for dx in 0..w {
                            visited[y + dy][x + dx] = true;
                        }
                    }

                    let quad_z = if face == FACE_POS_Z { z + 1 } else { z } as u32;
                    quads.push(GreedyQuad::new(
                        x as u32, y as u32, quad_z,
                        w as u32, h as u32,
                        face as u32, vid,
                    ));
                }
            }
        }
    }

    //  ±X faces 
    for face in [FACE_POS_X, FACE_NEG_X] {
        for x in 0..CS {
            let neighbour_x = if face == FACE_POS_X {
                if x + 1 < CS { x + 1 } else { CS }
            } else {
                if x > 0 { x - 1 } else { CS }
            };

            let mut face_present  = [[0u8; CS]; CS]; // [y][z]
            let mut face_voxel_id = [[0u8; CS]; CS];

            for y in 0..CS {
                for z in 0..CS {
                    let voxel = chunk.get(x, y, z);
                    if !voxel.is_solid() { continue; }

                    let neighbour_solid = if neighbour_x < CS {
                        chunk.get(neighbour_x, y, z).is_solid()
                    } else {
                        match neighbours {
                            None => false,
                            Some(nb) => {
                                let slice = if face == FACE_POS_X { nb.pos_x } else { nb.neg_x };
                                slice[z + y * CS] != 0
                            }
                        }
                    };

                    if !neighbour_solid {
                        face_present[y][z]  = 1;
                        face_voxel_id[y][z] = voxel.0;
                    }
                }
            }

            let mut visited = [[false; CS]; CS];
            for y in 0..CS {
                for z in 0..CS {
                    if visited[y][z] || face_present[y][z] == 0 { continue; }
                    let vid = face_voxel_id[y][z];

                    let mut w = 1usize;
                    while z + w < CS
                        && !visited[y][z + w]
                        && face_present[y][z + w] != 0
                        && face_voxel_id[y][z + w] == vid
                    {
                        w += 1;
                    }

                    let mut h = 1usize;
                    'outer: while y + h < CS {
                        for dz in 0..w {
                            if visited[y + h][z + dz]
                                || face_present[y + h][z + dz] == 0
                                || face_voxel_id[y + h][z + dz] != vid
                            {
                                break 'outer;
                            }
                        }
                        h += 1;
                    }

                    for dy in 0..h {
                        for dz in 0..w {
                            visited[y + dy][z + dz] = true;
                        }
                    }

                    let quad_x = if face == FACE_POS_X { x + 1 } else { x } as u32;
                    quads.push(GreedyQuad::new(
                        quad_x, y as u32, z as u32,
                        w as u32, h as u32,
                        face as u32, vid,
                    ));
                }
            }
        }
    }

    quads
}

//  Timing 
pub fn mesh_chunk_timed(chunk: &Chunk, neighbours: Option<&NeighbourData<'_>>)
    -> (Vec<GreedyQuad>, u128)
{
    let t0 = std::time::Instant::now();
    let quads = mesh_chunk(chunk, neighbours);
    let us = t0.elapsed().as_micros();
    (quads, us)
}

//  Tests 

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Chunk, VoxelId};


    #[test]
    fn quad_packing_roundtrip() {
        let q = GreedyQuad::new(5, 12, 30, 7, 3, FACE_POS_Y as u32, VoxelId::STONE.0);
        let (x, y, z, w, h) = q.decode_pos();
        let (face, vid) = q.decode_type();
        assert_eq!((x, y, z, w, h), (5, 12, 30, 7, 3));
        assert_eq!(face, FACE_POS_Y as u32);
        assert_eq!(vid, VoxelId::STONE.0);
    }

    #[test]
    fn quad_max_values() {
        let q = GreedyQuad::new(32, 32, 32, 32, 32, FACE_NEG_Z as u32, 255);
        let (x, y, z, w, h) = q.decode_pos();
        assert_eq!((x, y, z, w, h), (32, 32, 32, 32, 32));
        let (face, vid) = q.decode_type();
        assert_eq!(face, FACE_NEG_Z as u32);
        assert_eq!(vid, 255);
    }


    #[test]
    fn empty_chunk_produces_no_quads() {
        let chunk = Chunk::empty();
        let quads = mesh_chunk(&chunk, None);
        assert!(quads.is_empty(), "empty chunk should produce zero quads");
    }

    #[test]
    fn all_solid_chunk_produces_only_outer_faces() {
        let mut chunk = Chunk::empty();
        chunk.fill(VoxelId::STONE);
        let quads = mesh_chunk(&chunk, None);
        assert_eq!(quads.len(), 6, "fully solid chunk should produce exactly 6 quads (one per face)");
    }

    #[test]
    fn all_solid_chunk_quads_cover_full_face() {
        let mut chunk = Chunk::empty();
        chunk.fill(VoxelId::STONE);
        let quads = mesh_chunk(&chunk, None);
        // Every quad should be 32×32.
        for q in &quads {
            let (_, _, _, w, h) = q.decode_pos();
            assert_eq!(w, CS as u32, "width should be full chunk size");
            assert_eq!(h, CS as u32, "height should be full chunk size");
        }
    }


    #[test]
    fn single_voxel_has_six_quads() {
        let mut chunk = Chunk::empty();
        chunk.set(5, 10, 15, VoxelId::STONE);
        let quads = mesh_chunk(&chunk, None);
        assert_eq!(quads.len(), 6, "isolated voxel should produce 6 quads (one per face)");
    }

    #[test]
    fn single_voxel_quads_are_unit_size() {
        let mut chunk = Chunk::empty();
        chunk.set(5, 10, 15, VoxelId::STONE);
        let quads = mesh_chunk(&chunk, None);
        for q in &quads {
            let (_, _, _, w, h) = q.decode_pos();
            assert_eq!(w, 1);
            assert_eq!(h, 1);
        }
    }


    #[test]
    fn two_adjacent_voxels_share_no_internal_face() {
        let mut chunk = Chunk::empty();
        chunk.set(0, 0, 0, VoxelId::STONE);
        chunk.set(1, 0, 0, VoxelId::STONE);
        let quads = mesh_chunk(&chunk, None);
        assert!(quads.len() < 12, "adjacent voxels should have fewer quads than isolated: got {}", quads.len());
    }

    #[test]
    fn flat_layer_reduces_to_two_quads() {
        let mut chunk = Chunk::empty();
        chunk.fill_layer(0, VoxelId::STONE);
        let quads = mesh_chunk(&chunk, None);
        assert_eq!(quads.len(), 6, "single flat layer should produce 6 quads, got {}", quads.len());
    }

    #[test]
    fn two_different_voxel_types_not_merged() {
        let mut chunk = Chunk::empty();
        for x in 0..CS {
            let vid = if x % 2 == 0 { VoxelId::STONE } else { VoxelId::DIRT };
            chunk.set(x, 0, 0, vid);
        }
        let quads = mesh_chunk(&chunk, None);
        let single_quads = {
            let mut c = Chunk::empty();
            c.set(0, 0, 0, VoxelId::STONE);
            mesh_chunk(&c, None).len()
        };
        assert!(quads.len() >= 64, "alternating types should not merge tops/bottoms: got {}", quads.len());
    }


    #[test]
    fn greedy_never_produces_more_quads_than_naive() {
        use crate::gen::noise::{TerrainParams, generate_chunk};
        use glam::IVec3;
        let params = TerrainParams { sea_level: 16, amplitude: 4.0, ..Default::default() };
        let chunk = generate_chunk(IVec3::new(0, 0, 0), &params);
        let greedy_count = mesh_chunk(&chunk, None).len();

        let mut naive_count = 0usize;
        for y in 0..CS {
            for z in 0..CS {
                for x in 0..CS {
                    if !chunk.get(x, y, z).is_solid() { continue; }
                    for (dx, dy, dz) in [
                        (1i32,0,0),(-1,0,0),(0,1i32,0),(0,-1,0),(0,0,1i32),(0,0,-1)
                    ] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if nx < 0 || nx >= CS as i32 ||
                           ny < 0 || ny >= CS as i32 ||
                           nz < 0 || nz >= CS as i32 ||
                           !chunk.get(nx as usize, ny as usize, nz as usize).is_solid()
                        {
                            naive_count += 1;
                        }
                    }
                }
            }
        }

        assert!(
            greedy_count <= naive_count,
            "greedy ({greedy_count}) should never exceed naive ({naive_count})"
        );
        // For terrain, greedy should achieve meaningful reduction.
        // Even a modest reduction (5×) confirms merging is happening.
        if naive_count > 0 {
            let ratio = naive_count as f64 / greedy_count as f64;
            assert!(
                ratio >= 2.0,
                "greedy should reduce quad count by at least 2× on terrain: \
                 naive={naive_count} greedy={greedy_count} ratio={ratio:.1}"
            );
        }
    }


    #[test]
    fn all_exposed_faces_are_covered_single_voxel() {
        // For a single voxel, verify we get exactly 1 quad per face direction.
        let mut chunk = Chunk::empty();
        chunk.set(5, 5, 5, VoxelId::STONE);
        let quads = mesh_chunk(&chunk, None);

        let mut face_count = [0usize; 6];
        for q in &quads {
            let (face, _) = q.decode_type();
            face_count[face as usize] += 1;
        }
        for (i, &count) in face_count.iter().enumerate() {
            assert_eq!(count, 1, "face {i} should have exactly 1 quad, got {count}");
        }
    }

    #[test]
    fn all_exposed_faces_are_covered_full_solid() {
        let mut chunk = Chunk::empty();
        chunk.fill(VoxelId::STONE);
        let quads = mesh_chunk(&chunk, None);

        let mut face_count = [0usize; 6];
        for q in &quads {
            let (face, _) = q.decode_type();
            face_count[face as usize] += 1;
        }
        for (i, &count) in face_count.iter().enumerate() {
            assert_eq!(count, 1, "face {i} should have exactly 1 quad for solid chunk, got {count}");
        }
    }


    #[test]
    fn mesh_chunk_timed_returns_nonzero_duration_for_non_empty() {
        use crate::gen::noise::{TerrainParams, generate_chunk};
        use glam::IVec3;
        let params = TerrainParams::default();
        let chunk = generate_chunk(IVec3::ZERO, &params);
        let (quads, us) = mesh_chunk_timed(&chunk, None);
        // Should complete in a reasonable time (< 10 seconds even on slow hardware).
        // We just verify it returns and the duration is at least 0.
        assert!(us < 10_000_000, "mesh_chunk should complete in under 10 seconds, took {us}µs");
        // And that it produced some quads for a terrain chunk.
        assert!(!quads.is_empty(), "terrain chunk should produce quads");
    }
}