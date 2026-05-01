//! Optimised renderer entry point.
//! Build order: greedy mesher (1) → vertex pool (2) → pipeline/vert.wgsl (3)
//!              → indirect draw (4) → compute culling (5)

mod app;
mod indirect;
mod pipeline;
mod renderer;
mod vertex_pool;
mod world_manager;

use winit::event_loop::{ControlFlow, EventLoop};
use app::App;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::Uninitialized).expect("run event loop");
}