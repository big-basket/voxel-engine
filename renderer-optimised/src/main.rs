//! Optimised renderer entry point.
//! Build order: greedy mesher (1) → vertex pool (2) → pipeline/vert.wgsl (3)
//!              → indirect draw (4) → compute culling (5)

mod app;
mod bench;
mod cull_pipeline;
mod indirect;
mod overhead_breakdown;
mod pipeline;
mod renderer;
mod vertex_pool;
mod world_manager;

use winit::event_loop::{ControlFlow, EventLoop};
use app::App;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--bench") {
        bench::run_benchmarks();
        return;
    }

    if args.iter().any(|a| a == "--overhead") {
        overhead_breakdown::run_overhead_breakdown();
        return;
    }

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::Uninitialized).expect("run event loop");
}