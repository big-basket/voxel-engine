/// Indirect draw buffer for multi_draw_indirect.
use glam::IVec3;
use crate::vertex_pool::DrawIndirectArgs;

pub struct IndirectBuffer {
    pub buffer:     wgpu::Buffer,
    pub draw_count: u32,
    capacity:       u32,
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

        self.reset_buf = active_chunks.iter().map(|(_, a)| DrawIndirectArgs {
            vertex_count:   a.vertex_count,
            instance_count: 1,
            first_vertex:   a.first_vertex,
            first_instance: a.first_instance,
        }).collect();

        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.reset_buf));
        log::debug!("indirect buffer: {} draws", self.draw_count);
    }

    pub fn reset_instance_counts(&self, queue: &wgpu::Queue) {
        if self.reset_buf.is_empty() { return; }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.reset_buf));
    }
}