# Voxel Engine
A voxel rendering engine written in Rust, utilizing wgpu and winit.

## Project Structure
This project is organized as a Cargo workspace with three main components:
voxel-core: The core library handling shared world state, persistence (redb, rkyv), procedural terrain generation (noise), camera logic, and input handling. 
renderer-naive: A straightforward voxel rendering implementation.
renderer-optimised: An optimized rendering implementation.

## Controls
When running the engine, the mouse is captured automatically on your first left-click.
W, A, S, D: Move camera
Left Mouse Button (LMB): Remove Voxels
Right Mouse Button (RMB): Add Voxels 
Tab: Cycle block type for placing
[ / -: Decrease brush size
] / +: Increase brush size
Escape: Release mouse cursor

## Running the Project
To run the naive renderer, use the following command. Running in --release mode is highly recommended to ensure smooth performance during terrain generation and rendering.
cargo run -p renderer-naive --release

## Benchmarks
The project includes built-in benchmarking tools. You can run them by passing the --bench flag:
cargo run -p renderer-naive --release -- --bench


## License
MIT License (see Cargo.toml).
