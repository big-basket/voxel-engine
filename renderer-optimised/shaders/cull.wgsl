// Compute frustum culling pass.
//
// One thread per active draw entry in the indirect buffer.
// Each thread:
//   1. Reads the chunk origin from chunk_origins (indexed by first_instance / QUADS_PER_SLOT)
//   2. Builds the chunk AABB (origin to origin + CHUNK_SIZE)
//   3. Tests the AABB against the 6 camera frustum planes
//   4. Writes instance_count=0 to the indirect buffer entry if fully outside any plane
//      (no change if visible — instance_count was reset to 1 by the reset pass)
//
// This runs before the render pass each frame. The render pass is unchanged —
// multi_draw_indirect skips any entry with instance_count=0 automatically.

// ── Constants ─────────────────────────────────────────────────────────────────

const CHUNK_SIZE: f32 = 32.0;
const QUADS_PER_SLOT: u32 = 4096u;

// ── Uniforms ──────────────────────────────────────────────────────────────────

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec4<f32>,
    // 6 frustum planes in world space: vec4(normal.xyz, -dot(normal, point))
    // A point P is inside plane i if dot(plane.xyz, P) + plane.w >= 0
    frustum:   array<vec4<f32>, 6>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// ── Storage buffers ───────────────────────────────────────────────────────────

// Chunk world-space origins — one vec4 per pool slot.
// Indexed by first_instance / QUADS_PER_SLOT (same as vertex shader).
@group(1) @binding(0)
var<storage, read> chunk_origins: array<vec4<f32>>;

// Indirect draw buffer — DrawIndirectArgs layout:
//   [0] vertex_count   u32
//   [1] instance_count u32  ← cull writes 0 here to skip the draw
//   [2] first_vertex   u32
//   [3] first_instance u32  ← encodes the slot index × QUADS_PER_SLOT
struct DrawArgs {
    vertex_count:   u32,
    instance_count: u32,
    first_vertex:   u32,
    first_instance: u32,
}

@group(2) @binding(0)
var<storage, read_write> draw_args: array<DrawArgs>;

// ── Frustum test ──────────────────────────────────────────────────────────────

// Tests whether an AABB is fully outside any frustum plane.
// Returns true if the AABB should be CULLED (not drawn).
fn cull_aabb(min_p: vec3<f32>, max_p: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = camera.frustum[i];
        let n = plane.xyz;

        // Positive vertex: corner of AABB most in the direction of the plane normal.
        // If the positive vertex is outside (negative signed distance), the whole
        // AABB is outside and we can cull.
        let pos_x = select(min_p.x, max_p.x, n.x >= 0.0);
        let pos_y = select(min_p.y, max_p.y, n.y >= 0.0);
        let pos_z = select(min_p.z, max_p.z, n.z >= 0.0);

        let d = dot(n, vec3<f32>(pos_x, pos_y, pos_z)) + plane.w;
        if d < 0.0 {
            return true; // fully outside this plane → cull
        }
    }
    return false; // inside or intersecting all planes → keep
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let draw_index = gid.x;

    // Guard against overrun — dispatch size may be rounded up to workgroup size.
    if draw_index >= arrayLength(&draw_args) {
        return;
    }

    let args = draw_args[draw_index];

    // Skip already-zeroed entries (empty slots that snuck in).
    if args.vertex_count == 0u {
        return;
    }

    // Derive slot index from first_instance (= slot × QUADS_PER_SLOT).
    let slot_index = args.first_instance / QUADS_PER_SLOT;
    let origin = chunk_origins[slot_index].xyz;

    // Chunk AABB in world space.
    let aabb_min = origin;
    let aabb_max = origin + vec3<f32>(CHUNK_SIZE);

    if cull_aabb(aabb_min, aabb_max) {
        // Zero instance_count to skip this draw.
        draw_args[draw_index].instance_count = 0u;
    }
    // Visible chunks are left unchanged — the reset pass already set instance_count=1.
}