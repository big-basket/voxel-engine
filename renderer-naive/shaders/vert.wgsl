// Naive renderer vertex shader.
// Unpacks the two-u32 vertex format from mesh.rs and transforms to clip space.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec4<f32>,
    // 6 frustum planes — not used in vert shader but must match the uniform layout
    frustum:   array<vec4<f32>, 6>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Chunk world-space origin, passed as a push-constant-style uniform.
// We use a second bind group with a per-draw uniform rather than push constants
// because wgpu does not expose push constants on all backends.
struct ChunkUniform {
    origin: vec4<f32>, // xyz = chunk world origin, w unused
}

@group(1) @binding(0)
var<uniform> chunk: ChunkUniform;

struct VertexInput {
    @location(0) packed_pos:  u32,
    @location(1) packed_data: u32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) face:     u32,
    @location(1) voxel_id: u32,
    @location(2) world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Unpack local voxel position (0..32 each axis)
    let lx = f32( in.packed_pos        & 0x3Fu);
    let ly = f32((in.packed_pos >>  6u) & 0x3Fu);
    let lz = f32((in.packed_pos >> 12u) & 0x3Fu);

    // Unpack face index and voxel id
    let face     = (in.packed_data       ) & 0x7u;
    let voxel_id = (in.packed_data >>  3u) & 0xFFu;

    // World position = chunk origin + local position
    let world_pos = chunk.origin.xyz + vec3<f32>(lx, ly, lz);

    var out: VertexOutput;
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.face      = face;
    out.voxel_id  = voxel_id;
    out.world_pos = world_pos;
    return out;
}
