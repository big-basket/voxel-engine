//! Entry point — argument parsing and event loop creation only.
//! All application logic lives in `app.rs`.

mod app;
mod bench;
mod mesh;
mod pipeline;
mod renderer;
mod world_manager;

use winit::event_loop::{ControlFlow, EventLoop};
use app::App;

fn main() {
    env_logger::init();

    if std::env::args().any(|a| a == "--bench") {
        bench::run_benchmarks();
        return;
    }

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::Uninitialized).expect("run event loop");
}