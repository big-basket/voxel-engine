use glam::Vec3;
use super::camera::Camera;

/// Fly-cam movement speed and sensitivity constants.
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Movement speed in world units per second.
    pub move_speed: f32,
    /// Multiplier applied when the sprint key is held.
    pub sprint_multiplier: f32,
    /// Mouse look sensitivity (radians per pixel).
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

/// Yaw/pitch angles driving the camera look direction.
/// Roll is intentionally omitted — fly-cams without roll are more comfortable.
#[derive(Debug, Clone, Default)]
pub struct CameraController {
    /// Horizontal rotation in radians. Wraps at ±π.
    pub yaw: f32,
    /// Vertical rotation in radians. Clamped to ±89° to prevent gimbal flip.
    pub pitch: f32,
    pub config: ControllerConfig,
}

impl CameraController {
    pub fn new(config: ControllerConfig) -> Self {
        CameraController { config, ..Default::default() }
    }

    /// Applies a mouse delta (pixels) to yaw and pitch.
    pub fn apply_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * self.config.look_sensitivity;
        self.pitch -= dy * self.config.look_sensitivity;

        // Wrap yaw to [-π, π] to prevent float drift over long sessions.
        use std::f32::consts::PI;
        if self.yaw > PI { self.yaw -= 2.0 * PI; }
        if self.yaw < -PI { self.yaw += 2.0 * PI; }

        // Clamp pitch so the camera never flips upside-down.
        let max_pitch = 89.0_f32.to_radians();
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);
    }

    /// Computes the forward unit vector from current yaw and pitch.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        ).normalize()
    }

    /// Updates the camera's forward vector from the current yaw/pitch.
    pub fn update_camera_look(&self, camera: &mut Camera) {
        camera.forward = self.forward();
    }

    /// Moves the camera based on directional input for one frame.
    ///
    /// `axes` is a unit vector in camera-local space:
    ///   +X = strafe right, -X = strafe left
    ///   +Y = fly up,       -Y = fly down
    ///   +Z = move back,    -Z = move forward
    ///
    /// `dt` is the frame delta in seconds.
    /// `sprinting` applies the sprint multiplier.
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
        // Move forward (axes.z = -1)
        ctrl.apply_movement(&mut cam, Vec3::new(0.0, 0.0, -1.0), 1.0, false);
        // With yaw=0, pitch=0 forward=(1,0,0) so position moves in +X
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