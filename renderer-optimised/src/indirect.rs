/// Indirect draw buffer for multi_draw_indirect.
///
/// Contains only `DrawIndirectArgs` entries (16 bytes each) packed
/// contiguously so `multi_draw_indirect` reads them at the correct stride.
/// Chunk origins are stored separately in the renderer's `chunk_origins_buf`
/// storage buffer, indexed by slot number.

use glam::IVec3;

use crate::vertex_pool::DrawIndirectArgs;

pub struct IndirectBuffer {
    /// GPU buffer of packed `DrawIndirectArgs` — 16 bytes × draw_count.
    pub buffer: wgpu::Buffer,
    /// Number of active draw entries.
    pub draw_count: u32,
    capacity: u32,
}

impl IndirectBuffer {
    pub fn new(device: &wgpu::Device, max_chunks: u32) -> Self {
        let byte_size = max_chunks as u64
            * std::mem::size_of::<DrawIndirectArgs>() as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("indirect draw buffer"),
            size:               byte_size,
            usage:              wgpu::BufferUsages::INDIRECT
                              | wgpu::BufferUsages::COPY_DST
                              | wgpu::BufferUsages::STORAGE, // needed by compute cull pass
            mapped_at_creation: false,
        });
        IndirectBuffer { buffer, draw_count: 0, capacity: max_chunks }
    }

    /// Writes active DrawIndirectArgs contiguously into the buffer.
    /// Called after any chunk upload or remesh.
    pub fn rebuild(
        &mut self,
        queue: &wgpu::Queue,
        active_chunks: &[(IVec3, DrawIndirectArgs)],
    ) {
        self.draw_count = active_chunks.len() as u32;
        if active_chunks.is_empty() { return; }

        assert!(
            self.draw_count <= self.capacity,
            "IndirectBuffer overflow: {} > capacity {}", self.draw_count, self.capacity
        );

        let args: Vec<DrawIndirectArgs> = active_chunks.iter()
            .map(|(_, a)| *a)
            .collect();
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&args));
        log::debug!("indirect buffer: {} draws", self.draw_count);
    }
}

impl IndirectBuffer {
    /// Resets all active draw entries to instance_count=1 before the cull pass.
    pub fn reset_instance_counts(&self, queue: &wgpu::Queue) {
        const INSTANCE_COUNT_OFFSET: u64 = 4; // byte offset within DrawIndirectArgs
        const ARGS_SIZE: u64 = 16;             // size of DrawIndirectArgs
        let one: u32 = 1;
        for i in 0..self.draw_count as u64 {
            queue.write_buffer(
                &self.buffer,
                i * ARGS_SIZE + INSTANCE_COUNT_OFFSET,
                bytemuck::bytes_of(&one),
            );
        }
    }
}