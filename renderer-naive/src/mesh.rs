// Per-voxel vertex expansion.
//
// This module will expand a chunk's voxel grid into a flat vertex buffer
// where every visible face becomes two triangles (six vertices).
// It is stubbed here so main.rs compiles — filled in during the geometry phase.
//
// Expected interface once implemented:
// ```rust
// pub fn build_chunk_mesh(chunk: &Chunk, chunk_pos: IVec3) -> Vec<Vertex>
// ```