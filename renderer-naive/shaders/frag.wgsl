// Naive renderer fragment shader.
// VertexOutput is defined in vert.wgsl (concatenated before this file).

const FACE_POS_X: u32 = 0u;
const FACE_NEG_X: u32 = 1u;
const FACE_POS_Y: u32 = 2u;
const FACE_NEG_Y: u32 = 3u;
const FACE_POS_Z: u32 = 4u;
const FACE_NEG_Z: u32 = 5u;

fn palette_colour(id: u32) -> vec3<f32> {
    switch id {
        case 1u: { return vec3<f32>(0.50, 0.50, 0.50); } // stone
        case 2u: { return vec3<f32>(0.60, 0.40, 0.20); } // dirt
        case 3u: { return vec3<f32>(0.27, 0.55, 0.18); } // grass
        case 4u: { return vec3<f32>(0.93, 0.87, 0.55); } // sand
        case 5u: { return vec3<f32>(0.15, 0.45, 0.80); } // water
        default: { return vec3<f32>(1.00, 0.00, 1.00); } // magenta = unknown
    }
}

fn face_brightness(face: u32) -> f32 {
    switch face {
        case FACE_POS_Y: { return 1.00; }
        case FACE_NEG_Y: { return 0.50; }
        case FACE_POS_X, FACE_NEG_X: { return 0.75; }
        case FACE_POS_Z, FACE_NEG_Z: { return 0.80; }
        default: { return 0.70; }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base  = palette_colour(in.voxel_id);
    let light = face_brightness(in.face);
    return vec4<f32>(base * light, 1.0);
}
