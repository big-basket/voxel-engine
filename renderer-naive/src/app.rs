/// Application event loop state and input handling.
///
/// `App` implements winit's `ApplicationHandler` and owns all per-frame state:
/// the renderer, camera, controller, and input. `main.rs` creates the event
/// loop and hands control here — it contains no logic of its own.
use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use voxel_core::{
    camera::{Camera, CameraController, ControllerConfig},
    input::{InputState, Key},
};

use crate::renderer::NaiveRenderer;

// ── State ─────────────────────────────────────────────────────────────────────

pub enum App {
    Uninitialized,
    Running(RunningState),
}

pub struct RunningState {
    pub renderer:       NaiveRenderer,
    pub camera:         Camera,
    pub controller:     CameraController,
    pub input:          InputState,
    pub last_frame:     std::time::Instant,
    pub mouse_captured: bool,
}

// ── ApplicationHandler ────────────────────────────────────────────────────────

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if matches!(self, App::Running(_)) { return; }

        let window_attrs = Window::default_attributes()
            .with_title(
                "Voxel Engine — Naive Renderer"
            )
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));

        let window = Arc::new(
            event_loop.create_window(window_attrs).expect("create window"),
        );

        let size = window.inner_size();
        let renderer = match NaiveRenderer::new(window, size.width, size.height) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Renderer init failed: {e}");
                event_loop.exit();
                return;
            }
        };

        let mut camera = Camera::new(size.width as f32 / size.height as f32);
        camera.position = glam::Vec3::new(0.0, 80.0, 0.0);

        *self = App::Running(RunningState {
            renderer,
            camera,
            controller:     CameraController::new(ControllerConfig::default()),
            input:          InputState::new(),
            last_frame:     std::time::Instant::now(),
            mouse_captured: false,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let App::Running(state) = self else { return };

        match event {
            // ── Window lifecycle ──────────────────────────────────────────────
            WindowEvent::CloseRequested => {
                auto_save_if_dirty(state, "close");
                event_loop.exit();
            }

            // ── Keyboard ──────────────────────────────────────────────────────
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(key_code),
                    state: element_state,
                    ..
                },
                ..
            } => {
                let pressed = element_state == ElementState::Pressed;

                // Feed movement keys into InputState.
                if let Some(key) = map_key(key_code) {
                    if pressed { state.input.press(key); }
                    else       { state.input.release(key); }
                }

                if pressed {
                    handle_hotkey(key_code, state, event_loop);
                }
            }

            // ── Mouse buttons ─────────────────────────────────────────────────
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                let pressed = btn_state == ElementState::Pressed;
                log::debug!(
                    "mouse: {:?} pressed={} captured={}",
                    button, pressed, state.mouse_captured
                );
                handle_mouse_button(button, pressed, state);
            }

            // ── Resize ────────────────────────────────────────────────────────
            WindowEvent::Resized(new_size) => {
                state.renderer.resize(new_size.width, new_size.height);
                state.camera.set_aspect(new_size.width, new_size.height);
            }

            // ── Frame ─────────────────────────────────────────────────────────
            WindowEvent::RedrawRequested => {
                tick(state, event_loop);
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let App::Running(state) = self else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.mouse_captured {
                state.input.accumulate_mouse(dx as f32, dy as f32);
                let (dx, dy) = state.input.take_mouse_delta();
                state.controller.apply_mouse_delta(dx, dy);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let App::Running(state) = self {
            state.renderer.window.request_redraw();
        }
    }
}

// ── Per-frame tick ────────────────────────────────────────────────────────────

fn tick(state: &mut RunningState, event_loop: &ActiveEventLoop) {
    let now = std::time::Instant::now();
    let dt  = now.duration_since(state.last_frame).as_secs_f32();
    state.last_frame = now;

    let axes = state.input.movement_axes();
    state.controller.apply_movement(&mut state.camera, axes, dt, state.input.sprinting());
    state.controller.update_camera_look(&mut state.camera);

    match state.renderer.render(&state.camera) {
        Ok(()) => {}
        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
            let size = state.renderer.window.inner_size();
            state.renderer.resize(size.width, size.height);
        }
        Err(wgpu::SurfaceError::OutOfMemory) => {
            eprintln!("Out of GPU memory");
            event_loop.exit();
        }
        Err(e) => eprintln!("Surface error: {e}"),
    }

    state.renderer.window.request_redraw();
}

// ── Input handlers ────────────────────────────────────────────────────────────

fn handle_hotkey(
    key_code: KeyCode,
    state: &mut RunningState,
    event_loop: &ActiveEventLoop,
) {
    match key_code {
        KeyCode::Escape => {
            if state.mouse_captured {
                release_cursor(&state.renderer.window);
                state.mouse_captured = false;
            } else {
                auto_save_if_dirty(state, "Escape");
                event_loop.exit();
            }
        }
        KeyCode::F5 => {
            log::info!("F5: manual save");
            state.renderer.save();
        }
        KeyCode::Tab => {
            state.renderer.cycle_place_voxel();
        }
        KeyCode::BracketRight | KeyCode::Equal | KeyCode::NumpadAdd => {
            state.renderer.increase_brush();
        }
        KeyCode::BracketLeft | KeyCode::Minus | KeyCode::NumpadSubtract => {
            state.renderer.decrease_brush();
        }
        _ => {}
    }
}

fn handle_mouse_button(button: MouseButton, pressed: bool, state: &mut RunningState) {
    if !pressed { return; }

    match button {
        MouseButton::Left => {
            if !state.mouse_captured {
                log::info!("LMB: capturing cursor");
                capture_cursor(&state.renderer.window);
                state.mouse_captured = true;
            } else {
                log::info!("LMB: dig");
                if let Some(hit) = state.renderer.raycast(&state.camera) {
                    state.renderer.dig(&hit);
                } else {
                    log::info!("LMB: no hit");
                }
            }
        }
        MouseButton::Right => {
            if !state.mouse_captured {
                log::info!("RMB: cursor not captured — ignoring");
            } else {
                log::info!("RMB: place");
                if let Some(hit) = state.renderer.raycast(&state.camera) {
                    state.renderer.place(&hit);
                } else {
                    log::info!("RMB: no hit");
                }
            }
        }
        _ => {}
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn auto_save_if_dirty(state: &mut RunningState, trigger: &str) {
    let dirty = state.renderer.dirty_count();
    if dirty > 0 {
        log::info!("auto-save on {trigger}: {dirty} dirty chunk(s)");
        state.renderer.save();
    }
}

fn capture_cursor(window: &Window) {
    window.set_cursor_visible(false);
    if window.set_cursor_grab(CursorGrabMode::Confined).is_err() {
        let _ = window.set_cursor_grab(CursorGrabMode::Locked);
    }
}

fn release_cursor(window: &Window) {
    let _ = window.set_cursor_grab(CursorGrabMode::None);
    window.set_cursor_visible(true);
}

fn map_key(code: KeyCode) -> Option<Key> {
    match code {
        KeyCode::KeyW        => Some(Key::W),
        KeyCode::KeyA        => Some(Key::A),
        KeyCode::KeyS        => Some(Key::S),
        KeyCode::KeyD        => Some(Key::D),
        KeyCode::Space       => Some(Key::Space),
        KeyCode::ShiftLeft   => Some(Key::LShift),
        KeyCode::ControlLeft => Some(Key::LControl),
        _ => None,
    }
}