/// Returns the minimum buffer size aligned to wgpu's `COPY_BUFFER_ALIGNMENT`
/// (256 bytes). Uniform buffers must be a multiple of this size.
pub fn aligned_size(size: u64) -> u64 {
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    (size + align - 1) & !(align - 1)
}

/// Creates a wgpu buffer sized and labelled for a uniform struct `T`.
/// The buffer is `UNIFORM | COPY_DST` so `queue.write_buffer` works on it.
pub fn create_uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: aligned_size(std::mem::size_of::<T>() as u64),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Writes a `bytemuck::Pod` value into a uniform buffer.
pub fn write_uniform<T: bytemuck::Pod>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    value: &T,
) {
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(value));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_size_multiples_of_four() {
        // wgpu::COPY_BUFFER_ALIGNMENT == 4
        assert_eq!(aligned_size(0), 0);
        assert_eq!(aligned_size(1), 4);
        assert_eq!(aligned_size(3), 4);
        assert_eq!(aligned_size(4), 4);
        assert_eq!(aligned_size(5), 8);
        assert_eq!(aligned_size(176), 176); // already aligned
        assert_eq!(aligned_size(177), 180);
    }

    #[test]
    fn camera_uniform_is_already_aligned() {
        use crate::camera::CameraUniform;
        // CameraUniform is 176 bytes, which is divisible by 4.
        assert_eq!(CameraUniform::SIZE % wgpu::COPY_BUFFER_ALIGNMENT, 0);
        assert_eq!(aligned_size(CameraUniform::SIZE), CameraUniform::SIZE);
    }
}