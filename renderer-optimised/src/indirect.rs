/// Indirect draw buffer for multi_draw_indirect.
///
/// # Reset strategy
///
/// Each frame the cull shader zeroes instance_count for culled chunks.
/// Before the cull dispatch, instance_count must be restored to 1 for all
/// entries. The previous O(N) approach issued one write_buffer per chunk —
/// measured at 293µs for 400 chunks (19% of frame time). This version caches
/// a CPU-side reset_buf and writes the whole thing in one write_buffer call.

use glam::IVec3;
use crate::vertex_pool::DrawIndirectArgs;

pub struct IndirectBuffer {
    pub buffer:     wgpu::Buffer,
    pub draw_count: u32,
    capacity:       u32,
    /// CPU shadow with instance_count=1. Written whole to GPU each frame reset.
    reset_buf:      Vec<DrawIndirectArgs>,
}

impl IndirectBuffer {
    pub fn new(device: &wgpu::Device, max_chunks: u32) -> Self {
        let byte_size = max_chunks as u64 * std::mem::size_of::<DrawIndirectArgs>() as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("indirect draw buffer"),
            size:               byte_size,
            usage:              wgpu::BufferUsages::INDIRECT
                              | wgpu::BufferUsages::COPY_DST
                              | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        IndirectBuffer { buffer, draw_count: 0, capacity: max_chunks, reset_buf: Vec::new() }
    }

    /// Writes active DrawIndirectArgs into the GPU buffer and caches the
    /// reset_buf (instance_count=1 for every entry) for fast per-frame reset.
    pub fn rebuild(
        &mut self,
        queue: &wgpu::Queue,
        active_chunks: &[(IVec3, DrawIndirectArgs)],
    ) {
        self.draw_count = active_chunks.len() as u32;
        if active_chunks.is_empty() { self.reset_buf.clear(); return; }

        assert!(
            self.draw_count <= self.capacity,
            "IndirectBuffer overflow: {} > capacity {}", self.draw_count, self.capacity
        );

        // Cache reset_buf — same as args but instance_count forced to 1.
        self.reset_buf = active_chunks.iter().map(|(_, a)| DrawIndirectArgs {
            vertex_count:   a.vertex_count,
            instance_count: 1,
            first_vertex:   a.first_vertex,
            first_instance: a.first_instance,
        }).collect();

        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.reset_buf));
        log::debug!("indirect buffer: {} draws", self.draw_count);
    }

    /// Restores instance_count=1 for all active entries in ONE write_buffer call.
    /// O(1) driver overhead — previously was O(N) individual 4-byte writes.
    pub fn reset_instance_counts(&self, queue: &wgpu::Queue) {
        if self.reset_buf.is_empty() { return; }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.reset_buf));
    }
}