use std::collections::HashSet;

/// Logical key codes — a thin wrapper so we don't pull winit into tests.
/// The renderer's event loop maps winit KeyCode → this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    W, A, S, D,
    Space,      // fly up
    LShift,     // fly down
    LControl,   // sprint
}

/// Snapshot of keyboard and mouse state for one frame.
#[derive(Debug, Default)]
pub struct InputState {
    held: HashSet<Key>,
    /// Raw mouse delta accumulated since the last `take_mouse_delta` call.
    mouse_delta: (f32, f32),
    /// True if the left mouse button is currently pressed.
    pub lmb_pressed: bool,
    /// True if the right mouse button is currently pressed.
    pub rmb_pressed: bool,
}

impl InputState {
    pub fn new() -> Self {
        InputState::default()
    }

    // ── Key state ────────────────────────────────────────────────────────────

    pub fn press(&mut self, key: Key) {
        self.held.insert(key);
    }

    pub fn release(&mut self, key: Key) {
        self.held.remove(&key);
    }

    pub fn is_held(&self, key: Key) -> bool {
        self.held.contains(&key)
    }

    // ── Mouse ────────────────────────────────────────────────────────────────

    /// Accumulates a raw mouse motion event.
    pub fn accumulate_mouse(&mut self, dx: f32, dy: f32) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Returns the accumulated delta and resets it to zero.
    /// Call once per frame before passing to the camera controller.
    pub fn take_mouse_delta(&mut self) -> (f32, f32) {
        let delta = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        delta
    }

    // ── Movement axes ────────────────────────────────────────────────────────

    /// Returns a (non-normalised) movement axis vector from held keys.
    ///
    /// +X = right (D), -X = left (A)
    /// +Y = up (Space), -Y = down (LShift)
    /// +Z = backward (S), -Z = forward (W)  ← matches camera convention
    pub fn movement_axes(&self) -> glam::Vec3 {
        let x = self.is_held(Key::D) as i32 - self.is_held(Key::A) as i32;
        let y = self.is_held(Key::Space) as i32 - self.is_held(Key::LShift) as i32;
        let z = self.is_held(Key::S) as i32 - self.is_held(Key::W) as i32;
        glam::Vec3::new(x as f32, y as f32, z as f32)
    }

    /// True if the sprint modifier is held.
    pub fn sprinting(&self) -> bool {
        self.is_held(Key::LControl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn press_and_release() {
        let mut s = InputState::new();
        assert!(!s.is_held(Key::W));
        s.press(Key::W);
        assert!(s.is_held(Key::W));
        s.release(Key::W);
        assert!(!s.is_held(Key::W));
    }

    #[test]
    fn duplicate_press_is_idempotent() {
        let mut s = InputState::new();
        s.press(Key::W);
        s.press(Key::W);
        assert!(s.is_held(Key::W));
        s.release(Key::W);
        assert!(!s.is_held(Key::W));
    }

    #[test]
    fn release_unheld_key_does_not_panic() {
        let mut s = InputState::new();
        s.release(Key::A); // never pressed — should be a no-op
        assert!(!s.is_held(Key::A));
    }

    #[test]
    fn mouse_delta_accumulates() {
        let mut s = InputState::new();
        s.accumulate_mouse(3.0, -1.0);
        s.accumulate_mouse(2.0, 4.0);
        let (dx, dy) = s.take_mouse_delta();
        assert!((dx - 5.0).abs() < 1e-6);
        assert!((dy - 3.0).abs() < 1e-6);
    }

    #[test]
    fn take_mouse_delta_resets() {
        let mut s = InputState::new();
        s.accumulate_mouse(10.0, 10.0);
        s.take_mouse_delta();
        let (dx, dy) = s.take_mouse_delta();
        assert_eq!(dx, 0.0);
        assert_eq!(dy, 0.0);
    }

    #[test]
    fn movement_axes_forward() {
        let mut s = InputState::new();
        s.press(Key::W);
        let axes = s.movement_axes();
        assert_eq!(axes, Vec3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn movement_axes_backward() {
        let mut s = InputState::new();
        s.press(Key::S);
        assert_eq!(s.movement_axes(), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn movement_axes_strafe_right() {
        let mut s = InputState::new();
        s.press(Key::D);
        assert_eq!(s.movement_axes(), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn movement_axes_fly_up() {
        let mut s = InputState::new();
        s.press(Key::Space);
        assert_eq!(s.movement_axes(), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn movement_axes_opposing_keys_cancel() {
        let mut s = InputState::new();
        s.press(Key::W);
        s.press(Key::S);
        let axes = s.movement_axes();
        assert_eq!(axes.z, 0.0);
    }

    #[test]
    fn movement_axes_diagonal() {
        let mut s = InputState::new();
        s.press(Key::W);
        s.press(Key::D);
        let axes = s.movement_axes();
        assert_eq!(axes, Vec3::new(1.0, 0.0, -1.0));
    }

    #[test]
    fn no_keys_held_gives_zero_axes() {
        let s = InputState::new();
        assert_eq!(s.movement_axes(), Vec3::ZERO);
    }

    #[test]
    fn sprinting_flag() {
        let mut s = InputState::new();
        assert!(!s.sprinting());
        s.press(Key::LControl);
        assert!(s.sprinting());
        s.release(Key::LControl);
        assert!(!s.sprinting());
    }
}