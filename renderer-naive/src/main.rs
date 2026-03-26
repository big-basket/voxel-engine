use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use voxel_core::{
    camera::{Camera, CameraController, ControllerConfig},
    input::{InputState, Key},
};

mod renderer;
mod mesh;
mod pipeline;
mod bench;

use renderer::NaiveRenderer;

fn main() {
    env_logger::init();

    if std::env::args().any(|a| a == "--bench") {
        bench::run_benchmarks();
        return;
    }

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::Uninitialized;
    event_loop.run_app(&mut app).expect("run event loop");
}

// ── Application state ─────────────────────────────────────────────────────────

enum App {
    Uninitialized,
    Running(RunningState),
}

struct RunningState {
    renderer:       NaiveRenderer,
    camera:         Camera,
    controller:     CameraController,
    input:          InputState,
    last_frame:     std::time::Instant,
    mouse_captured: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if matches!(self, App::Running(_)) { return; }

        let window_attrs = Window::default_attributes()
            .with_title("Voxel Engine — Naive Renderer  |  LMB: dig  RMB: place  Tab: cycle block  Esc: release mouse  [-]/[+]: brush size")
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
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(key_code),
                    state: element_state,
                    ..
                },
                ..
            } => {
                let pressed = element_state == ElementState::Pressed;
                if let Some(key) = map_key(key_code) {
                    if pressed { state.input.press(key); }
                    else       { state.input.release(key); }
                }
                if pressed {
                    match key_code {
                        KeyCode::Escape => {
                            if state.mouse_captured {
                                release_cursor(&state.renderer.window);
                                state.mouse_captured = false;
                            } else {
                                event_loop.exit();
                            }
                        }
                        // Cycle the block type placed by right-click
                        KeyCode::Tab => {
                            state.renderer.cycle_place_voxel();
                        }
                        // Increase Brush Size
                        KeyCode::BracketRight | KeyCode::Equal | KeyCode::NumpadAdd => {
                            state.renderer.increase_brush();
                        }
                        // Decrease Brush Size
                        KeyCode::BracketLeft | KeyCode::Minus | KeyCode::NumpadSubtract => {
                            state.renderer.decrease_brush();
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                let pressed = btn_state == ElementState::Pressed;
                log::debug!("mouse button: {:?} pressed={} captured={}", button, pressed, state.mouse_captured);

                // Left-click: capture mouse on first click, dig on subsequent clicks
                if button == MouseButton::Left && pressed {
                    if !state.mouse_captured {
                        log::info!("LMB: capturing cursor");
                        capture_cursor(&state.renderer.window);
                        state.mouse_captured = true;
                    } else {
                        log::info!("LMB: firing raycast for dig");
                        if let Some(hit) = state.renderer.raycast(&state.camera) {
                            state.renderer.dig(&hit);
                        } else {
                            log::info!("LMB: no hit — nothing to dig");
                        }
                    }
                }

                // Right-click: place voxel (only when mouse is captured)
                if button == MouseButton::Right && pressed {
                    if !state.mouse_captured {
                        log::info!("RMB: mouse not captured yet — ignoring");
                    } else {
                        log::info!("RMB: firing raycast for place");
                        if let Some(hit) = state.renderer.raycast(&state.camera) {
                            state.renderer.place(&hit);
                        } else {
                            log::info!("RMB: no hit — nothing to place");
                        }
                    }
                }
            }

            WindowEvent::Resized(new_size) => {
                state.renderer.resize(new_size.width, new_size.height);
                state.camera.set_aspect(new_size.width, new_size.height);
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f32();
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

// ── Cursor capture ────────────────────────────────────────────────────────────

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

// ── Key mapping ───────────────────────────────────────────────────────────────

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