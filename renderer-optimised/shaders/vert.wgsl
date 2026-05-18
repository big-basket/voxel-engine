// Optimised renderer 

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec4<f32>,
    frustum:   array<vec4<f32>, 6>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;


struct ChunkOrigins {
    origins: array<vec4<f32>>,
}

@group(1) @binding(0)
var<storage, read> chunk_origins: ChunkOrigins;


struct GreedyQuad {
    pos_size:  u32,  
    face_type: u32,  
}

@group(2) @binding(0)
var<storage, read> quads: array<GreedyQuad>;

// Output

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) face:      u32,
    @location(1) voxel_id:  u32,
    @location(2) world_pos: vec3<f32>,
}



fn u_axis(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>(0.0, 0.0, 1.0); } // POS_X: u = +Z
        case 1u: { return vec3<f32>(0.0, 0.0, 1.0); } // NEG_X: u = +Z
        case 2u: { return vec3<f32>(1.0, 0.0, 0.0); } // POS_Y: u = +X
        case 3u: { return vec3<f32>(1.0, 0.0, 0.0); } // NEG_Y: u = +X
        case 4u: { return vec3<f32>(1.0, 0.0, 0.0); } // POS_Z: u = +X
        case 5u: { return vec3<f32>(1.0, 0.0, 0.0); } // NEG_Z: u = +X
        default: { return vec3<f32>(1.0, 0.0, 0.0); }
    }
}

fn v_axis(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>(0.0, 1.0, 0.0); } // POS_X: v = +Y
        case 1u: { return vec3<f32>(0.0, 1.0, 0.0); } // NEG_X: v = +Y
        case 2u: { return vec3<f32>(0.0, 0.0, 1.0); } // POS_Y: v = +Z
        case 3u: { return vec3<f32>(0.0, 0.0, 1.0); } // NEG_Y: v = +Z
        case 4u: { return vec3<f32>(0.0, 1.0, 0.0); } // POS_Z: v = +Y
        case 5u: { return vec3<f32>(0.0, 1.0, 0.0); } // NEG_Z: v = +Y
        default: { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

fn corner_u(corner: u32) -> f32 {
    switch corner {
        case 0u: { return 0.0; }
        case 1u: { return 1.0; }
        case 2u: { return 1.0; }
        case 3u: { return 0.0; }
        default: { return 0.0; }
    }
}

fn corner_v(corner: u32) -> f32 {
    switch corner {
        case 0u: { return 0.0; }
        case 1u: { return 0.0; }
        case 2u: { return 1.0; }
        case 3u: { return 1.0; }
        default: { return 0.0; }
    }
}

// Vertex shader

fn quad_corner_offset(face: u32, corner: u32, w: f32, h: f32) -> vec3<f32> {
    let u = u_axis(face);
    let v = v_axis(face);

    var us: f32;
    var vs_raw: f32;
    switch corner {
        case 0u: { us = 0.0; vs_raw = 0.0; }
        case 1u: { us =   w; vs_raw = 0.0; }
        case 2u: { us =   w; vs_raw =   h; }
        case 3u: { us = 0.0; vs_raw =   h; }
        default: { us = 0.0; vs_raw = 0.0; }
    }

    let needs_flip = face == 0u || face == 2u || face == 5u;
    let vs = select(vs_raw, h - vs_raw, needs_flip);
    return u * us + v * vs;
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) quad_base:      u32,
) -> VertexOutput {
    let local_vi   = vertex_index / 6u;
    let tri_vert   = vertex_index % 6u;
    let quad_index = quad_base + local_vi;

    var corner: u32;
    switch tri_vert {
        case 0u: { corner = 0u; }
        case 1u: { corner = 1u; }
        case 2u: { corner = 2u; }
        case 3u: { corner = 0u; }
        case 4u: { corner = 2u; }
        case 5u: { corner = 3u; }
        default: { corner = 0u; }
    }

    let q = quads[quad_index];

    // Unpack position and size (6 bits each).
    let ax = f32( q.pos_size        & 0x3Fu);
    let ay = f32((q.pos_size >>  6u) & 0x3Fu);
    let az = f32((q.pos_size >> 12u) & 0x3Fu);
    let aw = f32((q.pos_size >> 18u) & 0x3Fu);
    let ah = f32((q.pos_size >> 24u) & 0x3Fu);

    let face     = q.face_type & 0x7u;
    let voxel_id = (q.face_type >> 3u) & 0xFFu;

    let anchor    = vec3<f32>(ax, ay, az);
    let offset    = quad_corner_offset(face, corner, aw, ah);
    let slot_index = quad_base / 2048u;
    let world_origin = chunk_origins.origins[slot_index].xyz;
    let world_pos = world_origin + anchor + offset;

    var out: VertexOutput;
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.face      = face;
    out.voxel_id  = voxel_id;
    out.world_pos = world_pos;
    return out;
}


