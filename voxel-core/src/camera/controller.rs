use glam::Vec3;
use super::camera::Camera;

#[derive(Debug, Clone)]
pub struct ControllerConfig {
    pub move_speed: f32,
    pub sprint_multiplier: f32,
    pub look_sensitivity: f32,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        ControllerConfig {
            move_speed: 20.0,
            sprint_multiplier: 4.0,
            look_sensitivity: 0.002,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CameraController {
    pub yaw: f32,
    pub pitch: f32,
    pub config: ControllerConfig,
}

impl CameraController {
    pub fn new(config: ControllerConfig) -> Self {
        CameraController { config, ..Default::default() }
    }

    pub fn apply_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.config.look_sensitivity;
        self.pitch -= dy * self.config.look_sensitivity;

        use std::f32::consts::PI;
        if self.yaw > PI { self.yaw -= 2.0 * PI; }
        if self.yaw < -PI { self.yaw += 2.0 * PI; }

        let max_pitch = 89.0_f32.to_radians();
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        ).normalize()
    }

    pub fn update_camera_look(&self, camera: &mut Camera) {
        camera.forward = self.forward();
    }

  
    pub fn apply_movement(
        &self,
        camera: &mut Camera,
        axes: Vec3,
        dt: f32,
        sprinting: bool,
    ) {
        if axes == Vec3::ZERO {
            return;
        }

        let speed = self.config.move_speed
            * if sprinting { self.config.sprint_multiplier } else { 1.0 }
            * dt;

        let forward = self.forward();
        let right = forward.cross(Vec3::Y).normalize();
        let up = Vec3::Y;

        camera.position += right   * axes.x * speed;
        camera.position += up      * axes.y * speed;
        camera.position += forward * (-axes.z) * speed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn default_controller() -> CameraController {
        CameraController::new(ControllerConfig::default())
    }

    #[test]
    fn forward_at_zero_yaw_pitch_looks_pos_x() {
        // yaw=0, pitch=0 → forward = (cos0·cos0, sin0, sin0·cos0) = (1, 0, 0)
        let ctrl = default_controller();
        let fwd = ctrl.forward();
        assert!((fwd.x - 1.0).abs() < 1e-5, "fwd.x={}", fwd.x);
        assert!(fwd.y.abs() < 1e-5);
        assert!(fwd.z.abs() < 1e-5);
    }

    #[test]
    fn forward_is_unit_length() {
        let mut ctrl = default_controller();
        for yaw in [-PI, -FRAC_PI_2, 0.0, FRAC_PI_2, PI] {
            for pitch in [-1.5, -0.5, 0.0, 0.5, 1.5] {
                ctrl.yaw = yaw;
                ctrl.pitch = pitch;
                let len = ctrl.forward().length();
                assert!((len - 1.0).abs() < 1e-5, "yaw={yaw} pitch={pitch} len={len}");
            }
        }
    }

    #[test]
    fn pitch_clamp_prevents_flip() {
        let mut ctrl = default_controller();
        ctrl.apply_mouse_delta(0.0, -1_000_000.0); // huge upward drag
        assert!(ctrl.pitch <= 89.0_f32.to_radians() + 1e-5);

        ctrl.apply_mouse_delta(0.0, 1_000_000.0); // huge downward drag
        assert!(ctrl.pitch >= -89.0_f32.to_radians() - 1e-5);
    }

    #[test]
    fn yaw_wraps_within_pi() {
        let mut ctrl = default_controller();
        // Spin the camera around many times
        for _ in 0..1000 {
            ctrl.apply_mouse_delta(100.0, 0.0);
        }
        assert!(ctrl.yaw >= -PI - 1e-4 && ctrl.yaw <= PI + 1e-4,
            "yaw={} out of [-π, π]", ctrl.yaw);
    }

    #[test]
    fn mouse_delta_zero_does_not_change_angles() {
        let mut ctrl = default_controller();
        ctrl.yaw = 0.5;
        ctrl.pitch = 0.3;
        ctrl.apply_mouse_delta(0.0, 0.0);
        assert_eq!(ctrl.yaw, 0.5);
        assert_eq!(ctrl.pitch, 0.3);
    }

    #[test]
    fn movement_forward_advances_position() {
        let ctrl = default_controller();
        let mut cam = Camera::new(1.0);
        cam.position = Vec3::ZERO;
        ctrl.apply_movement(&mut cam, Vec3::new(0.0, 0.0, -1.0), 1.0, false);
        assert!(cam.position.x > 0.0, "camera should have moved forward");
    }

    #[test]
    fn movement_zero_axes_does_not_move() {
        let ctrl = default_controller();
        let mut cam = Camera::new(1.0);
        let start = cam.position;
        ctrl.apply_movement(&mut cam, Vec3::ZERO, 1.0, false);
        assert_eq!(cam.position, start);
    }

    #[test]
    fn sprint_moves_faster_than_walk() {
        let ctrl = default_controller();

        let mut walk_cam = Camera::new(1.0);
        walk_cam.position = Vec3::ZERO;
        ctrl.apply_movement(&mut walk_cam, Vec3::NEG_Z, 1.0, false);

        let mut sprint_cam = Camera::new(1.0);
        sprint_cam.position = Vec3::ZERO;
        ctrl.apply_movement(&mut sprint_cam, Vec3::NEG_Z, 1.0, true);

        let walk_dist = walk_cam.position.length();
        let sprint_dist = sprint_cam.position.length();
        assert!(sprint_dist > walk_dist, "sprint={sprint_dist} should > walk={walk_dist}");
    }

    #[test]
    fn update_camera_look_sets_forward() {
        let mut ctrl = default_controller();
        ctrl.yaw = FRAC_PI_2;
        ctrl.pitch = 0.0;
        let mut cam = Camera::new(1.0);
        ctrl.update_camera_look(&mut cam);
        let expected = ctrl.forward();
        assert!((cam.forward - expected).length() < 1e-5);
    }
}