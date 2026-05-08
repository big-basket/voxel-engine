use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use voxel_core::{
    benchmark::BenchmarkConfig,
    camera::{Camera, CameraController, ControllerConfig},
    input::{InputState, Key},
};

mod bench;
mod mesh;
mod pipeline;
mod renderer;
mod world_manager;

use renderer::NaiveRenderer;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--bench") {
        bench::run_benchmarks();
        return;
    }

    // --preview-scene <id>  opens the renderer positioned at the named scene's camera.
    // Lists available scenes if the id is not found.
    // Example: cargo run -p renderer-naive -- --preview-scene frustum_cull
    let scene_preview = args.windows(2)
        .find(|w| w[0] == "--preview-scene")
        .map(|w| w[1].clone());

    if let Some(ref id) = scene_preview {
        // Validate the scene ID exists and print its description.
        let config = BenchmarkConfig::load_or_default(
            std::path::Path::new("benchmark_config.json")
        );
        if let Some(scene) = config.scenes.iter().find(|s| s.id == *id) {
            log::info!("Preview: '{}' — {}", scene.id, scene.description);
        } else {
            eprintln!("Unknown scene: '{id}'");
            eprintln!("Available scenes:");
            for s in &config.scenes {
                eprintln!("  {} — {}", s.id, s.description);
            }
            std::process::exit(1);
        }
    }

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::Uninitialized { scene_preview })
        .expect("run event loop");
}

enum App {
    Uninitialized { scene_preview: Option<String> },
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
        let scene_preview = match self {
            App::Running(_) => return,
            App::Uninitialized { scene_preview } => scene_preview.take(),
        };

        // Resolve initial camera from scene preview if given.
        let preview_camera = scene_preview.as_deref().and_then(|id| {
            let config = BenchmarkConfig::load_or_default(
                std::path::Path::new("benchmark_config.json")
            );
            config.scenes.into_iter().find(|s| s.id == id).map(|s| {
                (s.camera_pos(), s.camera_forward())
            })
        });

        let title = if let Some(ref id) = scene_preview {
            format!(
                "Voxel Engine — Naive Renderer  [preview: {}]  \
                 WASD: move  Space/Shift: up/down  LClick: capture mouse  Esc: exit",
                id
            )
        } else {
            "Voxel Engine — Naive Renderer  \
             |  WASD: move  Space/Shift: up/down  LClick: capture mouse  Esc: exit"
                .into()
        };

        let window_attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));

        let window = Arc::new(
            event_loop.create_window(window_attrs).expect("create window"),
        );

        let size = window.inner_size();
        let renderer = match NaiveRenderer::new(window, size.width, size.height) {
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
                if pressed && key_code == KeyCode::Escape {
                    if state.mouse_captured {
                        release_cursor(&state.renderer.window);
                        state.mouse_captured = false;
                    } else {
                        event_loop.exit();
                    }
                }
            }

            WindowEvent::MouseInput {
                state: element_state,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                if element_state == ElementState::Pressed && !state.mouse_captured {
                    capture_cursor(&state.renderer.window);
                    state.mouse_captured = true;
                }
            }

            WindowEvent::Resized(new_size) => {
                state.renderer.resize(new_size.width, new_size.height);
                state.camera.set_aspect(new_size.width, new_size.height);
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt  = now.duration_since(state.last_frame).as_secs_f32();
                state.last_frame = now;

                let axes     = state.input.movement_axes();
                let sprinting = state.input.sprinting();
                state.controller.apply_movement(&mut state.camera, axes, dt, sprinting);
                state.controller.update_camera_look(&mut state.camera);

                match state.renderer.render(&state.camera) {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let s = state.renderer.window.inner_size();
                        state.renderer.resize(s.width, s.height);
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