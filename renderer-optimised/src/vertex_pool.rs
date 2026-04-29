/// GPU vertex pool — single persistent storage buffer for all chunk quads.
///
/// # Architecture
///
/// The naive renderer allocates one `wgpu::Buffer` per chunk. With thousands
/// of chunks this means thousands of bind group updates per frame. The vertex
/// pool instead maintains a single large `STORAGE` buffer that all chunks
/// share, indexed by byte offset. The GPU vertex shader reads directly from
/// this buffer using `vertex_index / 4` to find its quad.
///
/// # Slot allocator
///
/// The buffer is divided into fixed-size slots. A CPU-side free list tracks
/// which slots are available. When a chunk is meshed:
///   1. Allocate a slot (or multiple contiguous slots if the mesh is large)
///   2. Upload the quad data via `queue.write_buffer`
///   3. Record `(chunk_pos → slot_index, quad_count)` in the chunk table
///
/// When a chunk is dirtied and remeshed:
///   1. Free the old slot(s) back to the free list
///   2. Allocate new slot(s) and upload the new mesh
///
/// This is the "persistently mapped buffer" approach from Nick McDonald's
/// vertex pooling article, adapted for wgpu's ownership model.
///
/// # Slot sizing
///
/// Each slot holds `QUADS_PER_SLOT` quads × 8 bytes/quad = `SLOT_BYTES` bytes.
/// A worst-case surface chunk with ~3000 exposed faces fits in one slot.
/// Unusually dense chunks can occupy multiple contiguous slots.
///
/// # Indirect draw
///
/// Each occupied slot has a corresponding `DrawIndirectArgs` entry in a
/// parallel indirect buffer, updated whenever the slot changes. The render
/// pass calls `multi_draw_indirect` once, consuming all entries — N chunks
/// in one GPU call regardless of world size.

use std::collections::HashMap;

use glam::IVec3;

use bytemuck::Zeroable;

use voxel_core::gen::GreedyQuad;

// ── Sizing constants ──────────────────────────────────────────────────────────

/// Maximum quads a single slot can hold.
/// 4096 quads × 8 bytes = 32 KiB per slot — same size as the raw voxel data.
pub const QUADS_PER_SLOT: usize = 4096;

/// Byte size of one slot.
pub const SLOT_BYTES: u64 = (QUADS_PER_SLOT * std::mem::size_of::<GreedyQuad>()) as u64;

/// Maximum number of slots in the pool.
/// 2048 slots × 32 KiB = 64 MiB — comfortably fits in VRAM on the RX 6800M.
pub const MAX_SLOTS: usize = 2048;

/// Total pool buffer size in bytes.
pub const POOL_BYTES: u64 = SLOT_BYTES * MAX_SLOTS as u64;

// ── Slot handle ───────────────────────────────────────────────────────────────

/// A contiguous run of allocated slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotRange {
    /// Index of the first slot in the run.
    pub first: usize,
    /// Number of slots in the run.
    pub count: usize,
}

impl SlotRange {
    /// Byte offset of the first slot in the pool buffer.
    #[inline]
    pub fn byte_offset(&self) -> u64 {
        self.first as u64 * SLOT_BYTES
    }

    /// Total byte size of this range.
    #[inline]
    #[allow(dead_code)]
    pub fn byte_size(&self) -> u64 {
        self.count as u64 * SLOT_BYTES
    }

    /// Maximum number of quads that fit in this range.
    #[inline]
    #[allow(dead_code)]
    pub fn quad_capacity(&self) -> usize {
        self.count * QUADS_PER_SLOT
    }
}

// ── Free-list allocator ───────────────────────────────────────────────────────

/// A simple free-list slot allocator.
///
/// Maintains a `Vec<bool>` of slot occupancy (false = free).
/// Allocation scans for the first run of `n` contiguous free slots.
/// This is O(MAX_SLOTS) per allocation — acceptable because remeshing
/// is far less frequent than rendering.
pub struct SlotAllocator {
    /// `true` = slot is occupied.
    occupied: Vec<bool>,
    /// Total slots allocated (for metrics).
    pub allocated_count: usize,
}

impl SlotAllocator {
    pub fn new() -> Self {
        SlotAllocator {
            occupied: vec![false; MAX_SLOTS],
            allocated_count: 0,
        }
    }

    /// Allocates `n` contiguous free slots. Returns `None` if the pool is full.
    pub fn allocate(&mut self, n: usize) -> Option<SlotRange> {
        assert!(n > 0, "must allocate at least 1 slot");
        assert!(n <= MAX_SLOTS, "requested {n} slots exceeds MAX_SLOTS={MAX_SLOTS}");

        let mut run_start = 0;
        let mut run_len   = 0;

        for i in 0..MAX_SLOTS {
            if !self.occupied[i] {
                if run_len == 0 { run_start = i; }
                run_len += 1;
                if run_len == n {
                    for s in run_start..run_start + n {
                        self.occupied[s] = true;
                    }
                    self.allocated_count += n;
                    return Some(SlotRange { first: run_start, count: n });
                }
            } else {
                run_len = 0;
            }
        }
        None
    }

    /// Frees the slots in `range`, making them available for reuse.
    pub fn free(&mut self, range: SlotRange) {
        for s in range.first..range.first + range.count {
            debug_assert!(self.occupied[s], "slot {s} was already free");
            self.occupied[s] = false;
        }
        self.allocated_count -= range.count;
    }

    /// How many slots are currently free.
    pub fn free_count(&self) -> usize {
        MAX_SLOTS - self.allocated_count
    }

    /// How many slots are currently occupied.
    #[allow(dead_code)]
    pub fn used_count(&self) -> usize {
        self.allocated_count
    }
}

// ── Per-chunk record ──────────────────────────────────────────────────────────

/// What the vertex pool knows about one chunk.
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    /// Which slots hold this chunk's quads.
    pub slot_range: SlotRange,
    /// Number of quads actually written (may be < slot_range.quad_capacity()).
    pub quad_count: u32,
    /// Byte offset of this chunk's first quad in the pool buffer.
    #[allow(dead_code)]
    pub byte_offset: u64,
}

// ── Indirect draw args ────────────────────────────────────────────────────────

/// GPU indirect draw command — matches wgpu's `DrawIndirect` layout exactly.
/// `multi_draw_indirect` reads these from the indirect buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirectArgs {
    /// Number of vertices to draw (quad_count × 4).
    pub vertex_count:    u32,
    /// Number of instances (always 1 per chunk).
    pub instance_count:  u32,
    /// First vertex index in the pool buffer.
    pub first_vertex:    u32,
    /// First instance index (chunk index in the instance data buffer).
    pub first_instance:  u32,
}

// ── Vertex pool ───────────────────────────────────────────────────────────────

/// The GPU vertex pool — owns the storage buffer, slot allocator, chunk table,
/// and indirect draw buffer.
pub struct VertexPool {
    /// The main quad storage buffer (`STORAGE | COPY_DST`).
    pub quad_buffer: wgpu::Buffer,

    /// Parallel indirect draw buffer (`INDIRECT | COPY_DST`).
    /// Entry `i` corresponds to slot `i`.
    pub indirect_buffer: wgpu::Buffer,

    /// CPU-side slot allocator.
    pub(crate) allocator: SlotAllocator,

    /// Maps chunk position → its pool record.
    chunks: HashMap<IVec3, ChunkRecord>,

    /// Bind group for the quad storage buffer (used by the vertex shader).
    /// Created initially with the pool's own bgl; WorldManager replaces it
    /// with one created from the pipeline's bgl so group indices match.
    pub bind_group: wgpu::BindGroup,

    /// Bind group layout (needed when creating the pipeline).
    #[allow(dead_code)]
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl VertexPool {
    /// Creates the vertex pool and allocates GPU buffers.
    pub fn new(device: &wgpu::Device) -> Self {
        // Main quad storage buffer.
        // COPY_SRC is included so tests can read back data via a staging buffer.
        // In production the GPU only ever reads this via the storage binding.
        let quad_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex pool — quad storage"),
            size:  POOL_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_DST
                 | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Indirect draw buffer — one entry per slot (most will be zeroed/inactive)
        let indirect_size = (MAX_SLOTS * std::mem::size_of::<DrawIndirectArgs>()) as u64;
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex pool — indirect draw"),
            size:  indirect_size,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout for @group(2) @binding(0) in the vertex shader
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vertex pool bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty:               wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vertex pool bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: quad_buffer.as_entire_binding(),
            }],
        });

        VertexPool {
            quad_buffer,
            indirect_buffer,
            allocator: SlotAllocator::new(),
            chunks: HashMap::new(),
            bind_group,
            bind_group_layout,
        }
    }

    // ── Upload ────────────────────────────────────────────────────────────────

    /// Uploads a chunk's greedy quads into the pool.
    ///
    /// If the chunk already has a record, its old slots are freed first.
    /// Allocates the minimum number of slots needed for `quads`.
    ///
    /// Returns the `ChunkRecord` for the chunk, or `None` if the pool is full.
    pub fn upload_chunk(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        chunk_pos: IVec3,
        quads: &[GreedyQuad],
    ) -> Option<&ChunkRecord> {
        // Free any existing allocation for this chunk.
        if let Some(old) = self.chunks.remove(&chunk_pos) {
            self.allocator.free(old.slot_range);
            self.zero_indirect(queue, old.slot_range);
        }

        if quads.is_empty() {
            return None;
        }

        // How many slots do we need?
        let slots_needed = quads.len().div_ceil(QUADS_PER_SLOT);
        let slot_range   = self.allocator.allocate(slots_needed)?;

        // Write quads into the storage buffer.
        let byte_offset = slot_range.byte_offset();
        let data = bytemuck::cast_slice(quads);
        queue.write_buffer(&self.quad_buffer, byte_offset, data);

        // Write the indirect draw command for the first slot of this chunk.
        // `first_vertex` is the index of the first *vertex* (not quad):
        //   first_vertex = slot_first_quad_index × 4
        let first_quad   = slot_range.first * QUADS_PER_SLOT;
        let first_vertex = (first_quad * 4) as u32;
        let args = DrawIndirectArgs {
            vertex_count:   (quads.len() as u32) * 4,
            instance_count: 1,
            first_vertex,
            first_instance: slot_range.first as u32, // used as chunk index
        };
        let args_offset = (slot_range.first * std::mem::size_of::<DrawIndirectArgs>()) as u64;
        queue.write_buffer(&self.indirect_buffer, args_offset, bytemuck::bytes_of(&args));

        let record = ChunkRecord {
            slot_range,
            quad_count: quads.len() as u32,
            byte_offset,
        };
        self.chunks.insert(chunk_pos, record);
        self.chunks.get(&chunk_pos)
    }

    /// Removes a chunk from the pool entirely (e.g. chunk became all-air).
    pub fn remove_chunk(&mut self, queue: &wgpu::Queue, chunk_pos: &IVec3) {
        if let Some(record) = self.chunks.remove(chunk_pos) {
            self.allocator.free(record.slot_range);
            self.zero_indirect(queue, record.slot_range);
        }
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /// Returns the record for a chunk, if it is in the pool.
    #[allow(dead_code)]
    pub fn get_chunk(&self, chunk_pos: &IVec3) -> Option<&ChunkRecord> {
        self.chunks.get(chunk_pos)
    }

    /// Returns the total number of quads currently in the pool across all chunks.
    pub fn total_quads(&self) -> u64 {
        self.chunks.values().map(|r| r.quad_count as u64).sum()
    }

    /// Number of chunks currently in the pool.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Number of free slots remaining.
    #[allow(dead_code)]
    pub fn free_slots(&self) -> usize {
        self.allocator.free_count()
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Zeroes the indirect draw entries for the given slot range so the GPU
    /// skips them (instance_count = 0 = no draw).
    fn zero_indirect(&self, queue: &wgpu::Queue, range: SlotRange) {
        for s in range.first..range.first + range.count {
            let zero = DrawIndirectArgs::zeroed();
            let offset = (s * std::mem::size_of::<DrawIndirectArgs>()) as u64;
            queue.write_buffer(&self.indirect_buffer, offset, bytemuck::bytes_of(&zero));
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SlotAllocator ─────────────────────────────────────────────────────────

    #[test]
    fn allocate_single_slot() {
        let mut alloc = SlotAllocator::new();
        let r = alloc.allocate(1).expect("should succeed");
        assert_eq!(r.first, 0);
        assert_eq!(r.count, 1);
        assert_eq!(alloc.used_count(), 1);
        assert_eq!(alloc.free_count(), MAX_SLOTS - 1);
    }

    #[test]
    fn allocate_multiple_slots_are_contiguous() {
        let mut alloc = SlotAllocator::new();
        let r = alloc.allocate(5).expect("should succeed");
        assert_eq!(r.count, 5);
        assert_eq!(r.first + r.count - 1, r.first + 4);
        assert_eq!(alloc.used_count(), 5);
    }

    #[test]
    fn allocate_sequential_slots_do_not_overlap() {
        let mut alloc = SlotAllocator::new();
        let a = alloc.allocate(3).unwrap();
        let b = alloc.allocate(3).unwrap();
        assert_eq!(b.first, a.first + 3, "b should start where a ends");
    }

    #[test]
    fn free_returns_slots_for_reuse() {
        let mut alloc = SlotAllocator::new();
        let a = alloc.allocate(4).unwrap();
        let _b = alloc.allocate(4).unwrap();
        alloc.free(a);
        assert_eq!(alloc.used_count(), 4);

        // New allocation should reuse the freed slot at index 0.
        let c = alloc.allocate(2).unwrap();
        assert_eq!(c.first, 0, "should reuse freed slots");
    }

    #[test]
    fn allocate_fills_fragmented_pool() {
        let mut alloc = SlotAllocator::new();
        // Allocate all slots
        let all = alloc.allocate(MAX_SLOTS).expect("should fit");
        assert_eq!(all.first, 0);
        assert_eq!(all.count, MAX_SLOTS);
        // Pool is now full
        assert!(alloc.allocate(1).is_none(), "full pool should return None");
        // Free half, then re-allocate
        let half = SlotRange { first: 0, count: MAX_SLOTS / 2 };
        alloc.free(half);
        assert!(alloc.allocate(MAX_SLOTS / 2).is_some());
    }

    #[test]
    fn slot_range_byte_offset_correct() {
        let r = SlotRange { first: 3, count: 2 };
        assert_eq!(r.byte_offset(), 3 * SLOT_BYTES);
        assert_eq!(r.byte_size(),   2 * SLOT_BYTES);
        assert_eq!(r.quad_capacity(), 2 * QUADS_PER_SLOT);
    }

    #[test]
    fn draw_indirect_args_is_pod() {
        let args = DrawIndirectArgs {
            vertex_count: 12, instance_count: 1,
            first_vertex: 0,  first_instance: 0,
        };
        let bytes = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 16, "DrawIndirectArgs must be exactly 16 bytes");
    }

    // ── VertexPool GPU tests ──────────────────────────────────────────────────
    // These tests require a real GPU (or wgpu's Vulkan/DX12 backend) and read
    // data back via a staging buffer. They are skipped gracefully if no adapter
    // is available (e.g. in headless CI without a GPU).

    /// Creates a minimal wgpu device + queue for testing, or returns None.
    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            },
        ))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label:             Some("test device"),
                required_features: wgpu::Features::empty(),
                required_limits:   wgpu::Limits::default(),
                memory_hints:      wgpu::MemoryHints::default(),
            },
            None,
        )).ok()?;

        Some((device, queue))
    }

    /// Reads `byte_count` bytes back from a GPU buffer using a staging buffer.
    fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue,
                   src: &wgpu::Buffer, offset: u64, byte_count: u64) -> Vec<u8>
    {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size:  byte_count,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("readback") }
        );
        encoder.copy_buffer_to_buffer(src, offset, &staging, 0, byte_count);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");

        let data = slice.get_mapped_range().to_vec();
        drop(slice.get_mapped_range()); // must drop before unmap
        staging.unmap();
        data
    }

    /// Build N test quads with deterministic content.
    fn make_quads(n: usize) -> Vec<GreedyQuad> {
        (0..n).map(|i| {
            GreedyQuad::new(
                (i % 32) as u32,          // x
                ((i / 32) % 32) as u32,   // y
                0,                         // z
                1, 1,                      // w, h
                (i % 6) as u32,            // face
                ((i % 255) + 1) as u8,     // voxel_id (1..=255)
            )
        }).collect()
    }

    #[test]
    fn vertex_pool_write_and_readback_small() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter — skipping vertex pool GPU test");
            return;
        };

        let mut pool = VertexPool::new(&device);
        let quads = make_quads(8);
        let chunk_pos = IVec3::new(0, 0, 0);

        let record = pool.upload_chunk(&device, &queue, chunk_pos, &quads)
            .expect("upload should succeed");

        let byte_offset = record.byte_offset;
        let byte_count  = (quads.len() * std::mem::size_of::<GreedyQuad>()) as u64;

        queue.submit([]); // flush pending writes
        device.poll(wgpu::Maintain::Wait);

        let raw = read_buffer(&device, &queue, &pool.quad_buffer, byte_offset, byte_count);
        let read_back: &[GreedyQuad] = bytemuck::cast_slice(&raw);

        for (i, (expected, actual)) in quads.iter().zip(read_back.iter()).enumerate() {
            assert_eq!(
                expected, actual,
                "quad {i} mismatch: expected {expected:?}, got {actual:?}"
            );
        }

        assert_eq!(pool.chunk_count(), 1);
        assert_eq!(pool.total_quads(), 8);
    }

    #[test]
    fn vertex_pool_write_and_readback_full_slot() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter — skipping vertex pool GPU test");
            return;
        };

        let mut pool = VertexPool::new(&device);
        let quads = make_quads(QUADS_PER_SLOT); // exact slot capacity
        let chunk_pos = IVec3::new(1, 0, 0);

        pool.upload_chunk(&device, &queue, chunk_pos, &quads)
            .expect("full-slot upload should succeed");

        let record = pool.get_chunk(&chunk_pos).unwrap();
        assert_eq!(record.slot_range.count, 1, "should fit in one slot exactly");
        assert_eq!(record.quad_count as usize, QUADS_PER_SLOT);

        let byte_count = SLOT_BYTES;
        queue.submit([]);
        device.poll(wgpu::Maintain::Wait);

        let raw = read_buffer(&device, &queue,
            &pool.quad_buffer, record.byte_offset, byte_count);
        let read_back: &[GreedyQuad] = bytemuck::cast_slice(&raw);

        for (i, (e, a)) in quads.iter().zip(read_back.iter()).enumerate() {
            assert_eq!(e, a, "quad {i} mismatch");
        }
    }

    #[test]
    fn vertex_pool_overflow_allocates_extra_slot() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter — skipping vertex pool GPU test");
            return;
        };

        let mut pool = VertexPool::new(&device);
        let quads = make_quads(QUADS_PER_SLOT + 1); // needs 2 slots
        let chunk_pos = IVec3::new(2, 0, 0);

        let record = pool.upload_chunk(&device, &queue, chunk_pos, &quads)
            .expect("upload should succeed");
        assert_eq!(record.slot_range.count, 2, "should span 2 slots");
    }

    #[test]
    fn vertex_pool_remesh_frees_old_slot() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter — skipping vertex pool GPU test");
            return;
        };

        let mut pool = VertexPool::new(&device);
        let chunk_pos = IVec3::new(0, 0, 0);

        // First upload
        let q1 = make_quads(10);
        pool.upload_chunk(&device, &queue, chunk_pos, &q1).unwrap();
        let slots_after_first = pool.allocator.used_count();

        // Second upload (remesh)
        let q2 = make_quads(5);
        pool.upload_chunk(&device, &queue, chunk_pos, &q2).unwrap();
        let slots_after_second = pool.allocator.used_count();

        assert_eq!(slots_after_first, slots_after_second,
            "remesh should not grow slot count if new mesh fits in same number of slots");
        assert_eq!(pool.total_quads(), 5, "total quads should reflect new mesh");
    }

    #[test]
    fn vertex_pool_remove_chunk_frees_slots() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter — skipping vertex pool GPU test");
            return;
        };

        let mut pool = VertexPool::new(&device);
        let chunk_pos = IVec3::new(0, 0, 0);

        pool.upload_chunk(&device, &queue, chunk_pos, &make_quads(50)).unwrap();
        assert_eq!(pool.allocator.used_count(), 1);

        pool.remove_chunk(&queue, &chunk_pos);
        assert_eq!(pool.allocator.used_count(), 0);
        assert_eq!(pool.chunk_count(), 0);
        assert!(pool.get_chunk(&chunk_pos).is_none());
    }

    #[test]
    fn vertex_pool_multiple_chunks_correct_offsets() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter — skipping vertex pool GPU test");
            return;
        };

        let mut pool = VertexPool::new(&device);

        let positions = [
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(2, 0, 0),
        ];

        for &pos in &positions {
            let quads = make_quads(pos.x as usize * 10 + 5);
            pool.upload_chunk(&device, &queue, pos, &quads).unwrap();
        }

        queue.submit([]);
        device.poll(wgpu::Maintain::Wait);

        // Verify each chunk can be read back independently.
        for &pos in &positions {
            let record = pool.get_chunk(&pos).unwrap().clone();
            let expected = make_quads(pos.x as usize * 10 + 5);
            let byte_count = (expected.len() * std::mem::size_of::<GreedyQuad>()) as u64;

            let raw = read_buffer(&device, &queue,
                &pool.quad_buffer, record.byte_offset, byte_count);
            let read_back: &[GreedyQuad] = bytemuck::cast_slice(&raw);

            for (i, (e, a)) in expected.iter().zip(read_back.iter()).enumerate() {
                assert_eq!(e, a, "chunk {pos}: quad {i} mismatch");
            }
        }

        assert_eq!(pool.chunk_count(), 3);
    }
}