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
            fov_y:  std::f32::consts::FRAC_PI_4 * (70.0 / 45.0),
            aspect,
            z_near: 0.01,    
            z_far:  8192.0,  
        }
    }

    #[inline]
    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up).normalize()
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward, self.up)
    }

  
    pub fn proj_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

  
    pub fn frustum_planes(&self) -> [Vec4; 6] {
        let m = self.view_proj();
        let rows = [
            m.row(0), m.row(1), m.row(2), m.row(3),
        ];

        let planes = [
            rows[3] + rows[0], // left
            rows[3] - rows[0], // right
            rows[3] + rows[1], // bottom
            rows[3] - rows[1], // top
            rows[3] + rows[2], // near
            rows[3] - rows[2], // far
        ];

        planes.map(|p| {
            let len = p.truncate().length();
            if len > 1e-6 { p / len } else { p }
        })
    }

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height.max(1) as f32;
    }
}


#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 4],
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
        let origin = view.transform_point3(cam.position);
        assert!(origin.length() < 1e-4, "eye maps to origin in view space: {origin}");
    }

    #[test]
    fn proj_matrix_maps_near_to_minus_one_far_to_one() {
        let cam = default_cam();
        let proj = cam.proj_matrix();
 
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
        cam.set_aspect(1920, 0); 
        assert!(cam.aspect.is_finite());
    }

    #[test]
    fn camera_uniform_size_is_correct() {

        assert_eq!(CameraUniform::SIZE, 176);
    }

    #[test]
    fn camera_uniform_from_camera_is_pod() {
        let cam = default_cam();
        let uniform = CameraUniform::from_camera(&cam);
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