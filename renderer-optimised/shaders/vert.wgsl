// Optimised renderer — vertex pulling shader.
//
// Key difference from naive renderer:
//   NAIVE:  vertex buffer → input assembler → vs_main(packed_pos, packed_data)
//   OPTIMISED: no vertex buffer. The shader reads its own data from the quad
//              storage buffer using @builtin(vertex_index).
//
// Each quad occupies 4 consecutive vertex indices (one per corner).
//   quad_index  = vertex_index / 4
//   corner      = vertex_index % 4   (0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left)
//
// GreedyQuad layout (matches Rust struct in gen/greedy.rs):
//   pos_size  = x(6) | y(6) | z(6) | w(6) | h(6)
//   face_type = face(3) | voxel_id(8)

// ── Uniforms ──────────────────────────────────────────────────────────────────

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec4<f32>,
    frustum:   array<vec4<f32>, 6>,
}

struct ChunkOrigin {
    // xyz = world-space origin of this chunk, w unused.
    // One entry per chunk, indexed by instance_index.
    origin: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Per-chunk world-space origin — one uniform per draw call.
// Build order 4 (indirect) will replace this with an instance array.
@group(1) @binding(0)
var<uniform> chunk: ChunkOrigin;

// The quad storage buffer — shared across all chunks.
// @group(2) @binding(0) is used by the vertex pool's bind group.
struct GreedyQuad {
    pos_size:  u32,  // x(6) | y(6) | z(6) | w(6) | h(6)
    face_type: u32,  // face(3) | voxel_id(8)
}

@group(2) @binding(0)
var<storage, read> quads: array<GreedyQuad>;

// ── Output ────────────────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) face:      u32,
    @location(1) voxel_id:  u32,
    @location(2) world_pos: vec3<f32>,
}

// ── Corner and axis lookup functions ─────────────────────────────────────────
//
// WGSL does not allow indexing const arrays with runtime values.
// These switch-based functions are the portable equivalent.

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

// CCW corner offsets: (u_scale, v_scale) for corners 0..3
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

// ── Vertex shader ─────────────────────────────────────────────────────────────
//
// 6 vertices per quad (two triangles, no index buffer).
// Corners are numbered 0-3 on the face plane:
//   3 -- 2
//   |    |
//   0 -- 1
// Triangle 0: 0,1,2  Triangle 1: 0,2,3  → CCW when viewed from outside.
//
// For each face direction the U and V axes point in different world directions,
// so the same (u,v) parametric pattern produces consistent outward-facing CCW
// winding across all six face directions.

fn quad_corner_offset(face: u32, corner: u32, w: f32, h: f32) -> vec3<f32> {
    let u = u_axis(face);
    let v = v_axis(face);

    // Cross product check — u×v must equal the face outward normal for CCW winding.
    // Faces where u×v OPPOSES the normal need their v coordinate flipped.
    //
    // face 0 POS_X: u=-Z  v=+Y  u×v = (-Z)×(+Y) = +X  normal=+X  ✓
    // face 1 NEG_X: u=+Z  v=+Y  u×v = (+Z)×(+Y) = -X  normal=-X  ✓
    // face 2 POS_Y: u=+X  v=+Z  u×v = (+X)×(+Z) = -Y  normal=+Y  ✗ flip
    // face 3 NEG_Y: u=+X  v=+Z  u×v = (+X)×(+Z) = -Y  normal=-Y  ✓
    // face 4 POS_Z: u=+X  v=+Y  u×v = (+X)×(+Y) = +Z  normal=+Z  ✓
    // face 5 NEG_Z: u=-X  v=+Y  u×v = (-X)×(+Y) = -Z  normal=-Z  ✓
    //
    // Only POS_Y needs its v flipped.
    var us: f32;
    var vs_raw: f32;
    switch corner {
        case 0u: { us = 0.0; vs_raw = 0.0; }
        case 1u: { us =   w; vs_raw = 0.0; }
        case 2u: { us =   w; vs_raw =   h; }
        case 3u: { us = 0.0; vs_raw =   h; }
        default: { us = 0.0; vs_raw = 0.0; }
    }
    // Flip v for faces where u×v opposes the face normal:
    //   face 0 POS_X: u=+Z, v=+Y → u×v=-X, normal=+X → flip
    //   face 2 POS_Y: u=+X, v=+Z → u×v=-Y, normal=+Y → flip
    //   face 5 NEG_Z: u=+X, v=+Y → u×v=+Z, normal=-Z → flip
    let needs_flip = face == 0u || face == 2u || face == 5u;
    let vs = select(vs_raw, h - vs_raw, needs_flip);
    return u * us + v * vs;
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) quad_base:      u32,
) -> VertexOutput {
    // 6 vertices per quad: tri0 = corners 0,1,2 — tri1 = corners 0,2,3
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

    // Read the quad from the storage buffer.
    let q = quads[quad_index];

    // Unpack position and size (6 bits each).
    let ax = f32( q.pos_size        & 0x3Fu);
    let ay = f32((q.pos_size >>  6u) & 0x3Fu);
    let az = f32((q.pos_size >> 12u) & 0x3Fu);
    let aw = f32((q.pos_size >> 18u) & 0x3Fu);
    let ah = f32((q.pos_size >> 24u) & 0x3Fu);

    // Unpack face and voxel type.
    let face     = q.face_type & 0x7u;
    let voxel_id = (q.face_type >> 3u) & 0xFFu;

    // Compute corner world offset from anchor + face axes.
    let anchor    = vec3<f32>(ax, ay, az);
    let offset    = quad_corner_offset(face, corner, aw, ah);
    let world_pos = chunk.origin.xyz + anchor + offset;

    var out: VertexOutput;
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.face      = face;
    out.voxel_id  = voxel_id;
    out.world_pos = world_pos;
    return out;
}


