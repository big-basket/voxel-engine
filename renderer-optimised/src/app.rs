/// Application event loop for the optimised renderer.
/// Structure mirrors renderer-naive/src/app.rs exactly.
use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use voxel_core::{
    benchmark::BenchmarkConfig,
    benchmark::scenes::SceneKind,
    camera::{Camera, CameraController, ControllerConfig},
    input::{InputState, Key},
};

use crate::renderer::OptimisedRenderer;
use crate::world_manager::WorldExtent;

pub enum App {
    /// `scene_preview` — if Some, loads the named scene's camera position on startup.
    Uninitialized { scene_preview: Option<String> },
    Running(RunningState),
}

pub struct RunningState {
    pub renderer:       OptimisedRenderer,
    pub camera:         Camera,
    pub controller:     CameraController,
    pub input:          InputState,
    pub last_frame:     std::time::Instant,
    pub mouse_captured: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let scene_preview = match self {
            App::Running(_) => return,
            App::Uninitialized { scene_preview } => scene_preview.take(),
        };

        // Load the config once and find the requested scene (if any).
        // We need both the camera and the extent from the same scene, so we
        // find it once rather than searching twice.
        let config = BenchmarkConfig::load_or_default(
            std::path::Path::new("benchmark_config.json")
        );

        let found_scene = scene_preview.as_deref().and_then(|id| {
            config.scenes.into_iter().find(|s| s.id == id)
        });

        let preview_camera = found_scene.as_ref().map(|s| {
            log::info!("Preview: loading scene '{}'", s.id);
            log::info!("  {}", s.description);
            (s.camera_pos(), s.camera_forward())
        });

        let extent = found_scene
            .map(|s| {
                let (draw_radius, vertical_layers) = s.kind.spatial_extent();
                WorldExtent { draw_radius, vertical_layers, terrain: s.terrain.clone() }
            })
            .unwrap_or_default();
        log::info!(
            "WorldExtent: draw_radius={} vertical_layers={}",
            extent.draw_radius, extent.vertical_layers,
        );

        let title = if let Some(id) = &scene_preview {
            format!(
                "Voxel Engine — Optimised Renderer  [preview: {}]  \
                 LMB: dig  RMB: place  Tab: cycle block  \
                 [-]/[+]: brush  F5: save  Esc: release cursor", id
            )
        } else {
            "Voxel Engine — Optimised Renderer  \
             |  LMB: dig  RMB: place  Tab: cycle block  \
             [-]/[+]: brush  F5: save  Esc: release cursor".into()
        };

        let window_attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));

        let window = Arc::new(
            event_loop.create_window(window_attrs).expect("create window"),
        );

        let size = window.inner_size();
        let renderer = match OptimisedRenderer::new(window, size.width, size.height, extent) {
            Ok(r)  => r,
            Err(e) => {
                eprintln!("Renderer init failed: {e}");
                event_loop.exit();
                return;
            }
        };

        let mut camera = Camera::new(size.width as f32 / size.height as f32);
        if let Some((pos, fwd)) = preview_camera {
            camera.position = pos;
            camera.forward  = fwd;
        } else {
            camera.position = glam::Vec3::new(0.0, 80.0, 0.0);
        }

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
        &mut self, event_loop: &ActiveEventLoop,
        _window_id: WindowId, event: WindowEvent,
    ) {
        let App::Running(state) = self else { return };

        match event {
            WindowEvent::CloseRequested => {
                auto_save_if_dirty(state, "close");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(key_code),
                    state: element_state, ..
                }, ..
            } => {
                let pressed = element_state == ElementState::Pressed;
                if let Some(key) = map_key(key_code) {
                    if pressed { state.input.press(key); }
                    else       { state.input.release(key); }
                }
                if pressed { handle_hotkey(key_code, state, event_loop); }
            }
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                if btn_state == ElementState::Pressed {
                    handle_mouse_button(button, state);
                }
            }
            WindowEvent::Resized(sz) => {
                state.renderer.resize(sz.width, sz.height);
                state.camera.set_aspect(sz.width, sz.height);
            }
            WindowEvent::RedrawRequested => { tick(state, event_loop); }
            _ => {}
        }
    }

    fn device_event(
        &mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent,
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

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let App::Running(state) = self {
            state.renderer.window.request_redraw();
        }
    }
}

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
            let sz = state.renderer.window.inner_size();
            state.renderer.resize(sz.width, sz.height);
        }
        Err(wgpu::SurfaceError::OutOfMemory) => {
            eprintln!("Out of GPU memory");
            event_loop.exit();
        }
        Err(e) => eprintln!("Surface error: {e}"),
    }
    state.renderer.window.request_redraw();
}

fn handle_hotkey(key: KeyCode, state: &mut RunningState, event_loop: &ActiveEventLoop) {
    match key {
        KeyCode::Escape => {
            if state.mouse_captured {
                release_cursor(&state.renderer.window);
                state.mouse_captured = false;
            } else {
                auto_save_if_dirty(state, "Escape");
                event_loop.exit();
            }
        }
        KeyCode::F5 => { log::info!("F5: save"); state.renderer.save(); }
        KeyCode::Tab => { state.renderer.cycle_place_voxel(); }
        KeyCode::BracketRight | KeyCode::Equal | KeyCode::NumpadAdd      => { state.renderer.increase_brush(); }
        KeyCode::BracketLeft  | KeyCode::Minus | KeyCode::NumpadSubtract => { state.renderer.decrease_brush(); }
        _ => {}
    }
}

fn handle_mouse_button(button: MouseButton, state: &mut RunningState) {
    match button {
        MouseButton::Left => {
            if !state.mouse_captured {
                capture_cursor(&state.renderer.window);
                state.mouse_captured = true;
            } else if let Some(hit) = state.renderer.raycast(&state.camera) {
                state.renderer.dig(&hit);
            }
        }
        MouseButton::Right => {
            if state.mouse_captured {
                if let Some(hit) = state.renderer.raycast(&state.camera) {
                    state.renderer.place(&hit);
                }
            }
        }
        _ => {}
    }
}

fn auto_save_if_dirty(state: &mut RunningState, trigger: &str) {
    let n = state.renderer.dirty_count();
    if n > 0 { log::info!("auto-save on {trigger}: {n} chunk(s)"); state.renderer.save(); }
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