//! `voxel-core` — shared world state, persistence, terrain generation,
//! camera, input, and benchmarking for the voxel rendering engine.
//!
//! Both `renderer-naive` and `renderer-optimised` depend on this crate.
//! All unit tests live here; the renderer crates contain no tests of their own.

pub mod benchmark;
pub mod camera;
pub mod gen;
pub mod gpu;
pub mod input;
pub mod persistence;
pub mod world;