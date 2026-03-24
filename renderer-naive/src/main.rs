use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use voxel_core::{
    camera::{Camera, CameraController, ControllerConfig},
    gpu::{GpuContext, GpuError},
    input::{InputState, Key},
};

mod renderer;
mod mesh;
mod pipeline;

use renderer::NaiveRenderer;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::Uninitialized;
    event_loop.run_app(&mut app).expect("run event loop");
}

// ── Application state machine ─────────────────────────────────────────────────
// winit 0.30 uses an ApplicationHandler trait where the window is created
// inside `resumed()` rather than before the event loop starts.

enum App {
    Uninitialized,
    Running(RunningState),
}

struct RunningState {
    renderer: NaiveRenderer,
    camera: Camera,
    controller: CameraController,
    input: InputState,
    last_frame: std::time::Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if matches!(self, App::Running(_)) {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("Voxel Engine — Naive Renderer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));

        let window = Arc::new(
            event_loop.create_window(window_attrs).expect("create window"),
        );

        let gpu = match GpuContext::new(wgpu::Features::empty()) {
            Ok(g) => g,
            Err(GpuError::NoAdapter) => {
                eprintln!("No GPU adapter found — cannot start renderer.");
                event_loop.exit();
                return;
            }
            Err(e) => {
                eprintln!("GPU init error: {e}");
                event_loop.exit();
                return;
            }
        };

        log::info!("GPU: {}", gpu.adapter_info());

        let size = window.inner_size();
        let renderer = NaiveRenderer::new(gpu, window, size.width, size.height);

        let camera = Camera::new(size.width as f32 / size.height as f32);
        let controller = CameraController::new(ControllerConfig::default());

        *self = App::Running(RunningState {
            renderer,
            camera,
            controller,
            input: InputState::new(),
            last_frame: std::time::Instant::now(),
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
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

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
                // Escape exits
                if key_code == KeyCode::Escape && pressed {
                    event_loop.exit();
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

                // Update camera from input
                let axes = state.input.movement_axes();
                let sprinting = state.input.sprinting();
                state.controller.apply_movement(&mut state.camera, axes, dt, sprinting);
                state.controller.update_camera_look(&mut state.camera);

                // Render
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
            state.input.accumulate_mouse(dx as f32, dy as f32);
            let (dx, dy) = state.input.take_mouse_delta();
            state.controller.apply_mouse_delta(dx, dy);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let App::Running(state) = self {
            state.renderer.window.request_redraw();
        }
    }
}

// ── Key mapping ───────────────────────────────────────────────────────────────

fn map_key(code: KeyCode) -> Option<Key> {
    match code {
        KeyCode::KeyW      => Some(Key::W),
        KeyCode::KeyA      => Some(Key::A),
        KeyCode::KeyS      => Some(Key::S),
        KeyCode::KeyD      => Some(Key::D),
        KeyCode::Space     => Some(Key::Space),
        KeyCode::ShiftLeft => Some(Key::LShift),
        KeyCode::ControlLeft => Some(Key::LControl),
        _ => None,
    }
}