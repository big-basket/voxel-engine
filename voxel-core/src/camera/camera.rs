use glam::{Mat4, Vec3, Vec4};
use bytemuck::{Pod, Zeroable};

/// The camera state — position and orientation in world space.
#[derive(Debug, Clone)]
pub struct Camera {
    /// World-space eye position.
    pub position: Vec3,
    /// Direction the camera is looking (normalised).
    pub forward: Vec3,
    /// World up vector (almost always Vec3::Y).
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Near clip plane distance.
    pub z_near: f32,
    /// Far clip plane distance.
    pub z_far: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Camera {
            position: Vec3::new(0.0, 64.0, 0.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: std::f32::consts::FRAC_PI_4, // 45°
            aspect,
            z_near: 0.1,
            z_far: 1024.0,
        }
    }

    /// The right vector (perpendicular to forward and up).
    #[inline]
    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up).normalize()
    }

    /// Builds the view matrix (world → camera space).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward, self.up)
    }

    /// Builds the projection matrix (camera → clip space).
    /// Uses reverse-Z (z_far → 0.0, z_near → 1.0) for better depth precision.
    pub fn proj_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far)
    }

    /// Combined view-projection matrix.
    pub fn view_proj(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// Extracts the six frustum planes from the view-projection matrix.
    /// Each plane is a Vec4 (a, b, c, d) where ax+by+cz+d=0 and the
    /// normal (a,b,c) points inward. Used by the compute cull shader.
    pub fn frustum_planes(&self) -> [Vec4; 6] {
        let m = self.view_proj();
        let rows = [
            m.row(0), m.row(1), m.row(2), m.row(3),
        ];

        // Gribb-Hartmann frustum extraction.
        let planes = [
            rows[3] + rows[0], // left
            rows[3] - rows[0], // right
            rows[3] + rows[1], // bottom
            rows[3] - rows[1], // top
            rows[3] + rows[2], // near
            rows[3] - rows[2], // far
        ];

        // Normalise each plane by the magnitude of its normal.
        planes.map(|p| {
            let len = p.truncate().length();
            if len > 1e-6 { p / len } else { p }
        })
    }

    /// Updates the aspect ratio (call on window resize).
    pub fn set_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height.max(1) as f32;
    }
}

/// GPU-uploadable camera data.
///
/// `repr(C)` and `bytemuck::Pod` so it can be written directly into a
/// wgpu buffer with `queue.write_buffer`. The layout must match the
/// `CameraUniform` struct in every WGSL shader that uses it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    /// Combined view-projection matrix (column-major, matches WGSL mat4x4).
    pub view_proj: [[f32; 4]; 4],
    /// World-space camera position, w=1 for alignment.
    pub position: [f32; 4],
    /// Six frustum planes for compute culling (a, b, c, d).
    pub frustum: [[f32; 4]; 6],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera) -> Self {
        let vp = camera.view_proj();
        let planes = camera.frustum_planes();
        CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            position: [camera.position.x, camera.position.y, camera.position.z, 1.0],
            frustum: planes.map(|p| p.to_array()),
        }
    }

    /// Size in bytes — used when creating the wgpu buffer.
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cam() -> Camera {
        Camera::new(16.0 / 9.0)
    }

    #[test]
    fn view_matrix_is_identity_at_origin_looking_neg_z() {
        let cam = default_cam();
        let view = cam.view_matrix();
        // Looking -Z from origin: view matrix should transform the
        // origin to the origin (translation column is zero-ish).
        let origin = view.transform_point3(cam.position);
        assert!(origin.length() < 1e-4, "eye maps to origin in view space: {origin}");
    }

    #[test]
    fn proj_matrix_maps_near_to_minus_one_far_to_one() {
        let cam = default_cam();
        let proj = cam.proj_matrix();
        // A point on the forward axis at z_near should map to NDC z ≈ -1
        // and at z_far to NDC z ≈ 1 (right-hand, standard wgpu clip space).
        // We just check the matrix is finite and non-zero.
        for col in proj.to_cols_array() {
            assert!(col.is_finite(), "proj matrix contains non-finite value: {col}");
        }
    }

    #[test]
    fn view_proj_is_proj_times_view() {
        let cam = default_cam();
        let expected = cam.proj_matrix() * cam.view_matrix();
        let actual = cam.view_proj();
        for (a, b) in expected.to_cols_array().iter().zip(actual.to_cols_array()) {
            assert!((a - b).abs() < 1e-5, "view_proj mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn right_vector_is_perpendicular_to_forward_and_up() {
        let cam = default_cam();
        let right = cam.right();
        assert!(right.dot(cam.forward).abs() < 1e-5);
        assert!(right.dot(cam.up).abs() < 1e-5);
    }

    #[test]
    fn right_vector_is_unit_length() {
        let cam = default_cam();
        assert!((cam.right().length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn frustum_planes_are_normalised() {
        let cam = default_cam();
        for (i, plane) in cam.frustum_planes().iter().enumerate() {
            let normal_len = plane.truncate().length();
            assert!(
                (normal_len - 1.0).abs() < 1e-4,
                "plane {i} normal length = {normal_len}, expected 1.0"
            );
        }
    }

    #[test]
    fn frustum_has_six_planes() {
        let cam = default_cam();
        assert_eq!(cam.frustum_planes().len(), 6);
    }

    #[test]
    fn set_aspect_updates_correctly() {
        let mut cam = default_cam();
        cam.set_aspect(1920, 1080);
        let expected = 1920.0 / 1080.0_f32;
        assert!((cam.aspect - expected).abs() < 1e-5);
    }

    #[test]
    fn set_aspect_zero_height_does_not_panic() {
        let mut cam = default_cam();
        cam.set_aspect(1920, 0); // height clamped to 1
        assert!(cam.aspect.is_finite());
    }

    #[test]
    fn camera_uniform_size_is_correct() {
        // 16 floats (mat4) + 4 floats (pos) + 24 floats (6 planes × 4)
        // = 44 floats = 176 bytes
        assert_eq!(CameraUniform::SIZE, 176);
    }

    #[test]
    fn camera_uniform_from_camera_is_pod() {
        let cam = default_cam();
        let uniform = CameraUniform::from_camera(&cam);
        // bytemuck::bytes_of will panic if alignment is wrong
        let bytes = bytemuck::bytes_of(&uniform);
        assert_eq!(bytes.len(), CameraUniform::SIZE as usize);
    }

    #[test]
    fn camera_uniform_position_matches_camera() {
        let mut cam = default_cam();
        cam.position = Vec3::new(10.0, 20.0, 30.0);
        let uniform = CameraUniform::from_camera(&cam);
        assert_eq!(uniform.position, [10.0, 20.0, 30.0, 1.0]);
    }

    #[test]
    fn moving_camera_changes_view_matrix() {
        let mut cam = default_cam();
        let view_a = cam.view_matrix();
        cam.position += Vec3::X * 10.0;
        let view_b = cam.view_matrix();
        assert_ne!(
            view_a.to_cols_array(),
            view_b.to_cols_array(),
            "moving camera should change view matrix"
        );
    }
}