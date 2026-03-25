struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec4<f32>,
    frustum:   array<vec4<f32>, 6>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct ChunkUniform {
    origin: vec4<f32>,
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
    // 6-bit per axis to match mesh.rs packing (<< 6, << 12)
    let lx = f32( in.packed_pos        & 0x3Fu);
    let ly = f32((in.packed_pos >>  6u) & 0x3Fu);
    let lz = f32((in.packed_pos >> 12u) & 0x3Fu);

    let face     = (in.packed_data      ) & 0x7u;
    let voxel_id = (in.packed_data >> 3u) & 0xFFu;

    let world_pos = chunk.origin.xyz + vec3<f32>(lx, ly, lz);

    var out: VertexOutput;
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.face      = face;
    out.voxel_id  = voxel_id;
    out.world_pos = world_pos;
    return out;
}
