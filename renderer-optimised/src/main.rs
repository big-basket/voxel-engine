//! Optimised renderer — entry point.
//! Vertex pool, indirect draw, and compute culling pipeline.
//! Build order: vertex_pool (2) → pipeline (3) → indirect (4) → cull shader (5).

mod vertex_pool;

fn main() {
    env_logger::init();
    println!("renderer-optimised: not yet implemented — run renderer-naive for now");
}