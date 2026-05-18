// Compute frustum culling pass

const CHUNK_SIZE: f32 = 32.0;
// MUST match QUADS_PER_SLOT in vertex_pool.rs.
const QUADS_PER_SLOT: u32 = 2048u;

// Uniforms 

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec4<f32>,

    frustum:   array<vec4<f32>, 6>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Storage buffers
@group(1) @binding(0)
var<storage, read> chunk_origins: array<vec4<f32>>;

struct DrawArgs {
    vertex_count:   u32,
    instance_count: u32,
    first_vertex:   u32,
    first_instance: u32,
}

@group(2) @binding(0)
var<storage, read_write> draw_args: array<DrawArgs>;

// Frustum test 

// Tests whether an AABB is fully outside any frustum plane
fn cull_aabb(min_p: vec3<f32>, max_p: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = camera.frustum[i];
        let n = plane.xyz;

        let pos_x = select(min_p.x, max_p.x, n.x >= 0.0);
        let pos_y = select(min_p.y, max_p.y, n.y >= 0.0);
        let pos_z = select(min_p.z, max_p.z, n.z >= 0.0);

        let d = dot(n, vec3<f32>(pos_x, pos_y, pos_z)) + plane.w;
        if d < 0.0 {
            return true; 
        }
    }
    return false; 
}

// Main

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let draw_index = gid.x;

    if draw_index >= arrayLength(&draw_args) {
        return;
    }

    let args = draw_args[draw_index];

    if args.vertex_count == 0u {
        return;
    }

    let slot_index = args.first_instance / QUADS_PER_SLOT;
    let origin = chunk_origins[slot_index].xyz;

    let aabb_min = origin;
    let aabb_max = origin + vec3<f32>(CHUNK_SIZE);

    if cull_aabb(aabb_min, aabb_max) {
        // Zero instance_count to skip this draw.
        draw_args[draw_index].instance_count = 0u;
    }
}