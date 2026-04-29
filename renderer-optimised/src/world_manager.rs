/// WorldManager — greedy meshing, vertex pool upload, persistence.
/// Mirrors renderer-naive's world_manager.rs but uses GreedyQuad + VertexPool
/// instead of per-chunk vertex buffers.
use std::collections::HashSet;

use glam::IVec3;
use wgpu::util::DeviceExt;

use voxel_core::{
    gen::{TerrainParams, generate_chunk, mesh_chunk},
    persistence::ChunkStore,
    world::{VoxelId, World, CHUNK_SIZE_I, chunk_pos_of},
    camera::Camera,
    input::{RayHit, place, raycast, remove},
};

use crate::pipeline::{ChunkOriginUniform, OptimisedPipeline};
use crate::vertex_pool::VertexPool;

// ── Per-chunk CPU record ──────────────────────────────────────────────────────

/// Everything the renderer needs to issue one draw call for a chunk.
pub struct ChunkDraw {
    /// World-space origin uniform buffer (one per chunk for build order 3).
    /// Build order 4 replaces this with an instance array.
    #[allow(dead_code)]
    pub origin_buf:        wgpu::Buffer,
    pub origin_bind_group: wgpu::BindGroup,
    /// Number of vertices (quad_count × 4) for the draw call.
    pub vertex_count:      u32,
    /// First quad index in the pool — passed as instance_index to the shader.
    pub first_quad:        u32,
}

// ── WorldManager ──────────────────────────────────────────────────────────────

pub struct WorldManager {
    pub world:        World,
    pub vertex_pool:  VertexPool,
    pub chunk_draws:  std::collections::HashMap<IVec3, ChunkDraw>,

    store:        ChunkStore,

    pub place_voxel:   VoxelId,
    pub reach:         f32,
    pub brush_radius:  u32,
}

impl WorldManager {
    const SAVE_PATH: &'static str = "world_optimised.db";

    pub fn new(device: &wgpu::Device, pipeline: &OptimisedPipeline) -> Self {
        let store = match ChunkStore::open(Self::SAVE_PATH) {
            Ok(s)  => { log::info!("persistence: opened '{}'", Self::SAVE_PATH); s }
            Err(e) => {
                log::warn!("persistence: could not open store ({e}) — edits not saved");
                ChunkStore::open(":memory:").expect("in-memory fallback")
            }
        };

        let mut vertex_pool = VertexPool::new(device);

        // Temporary: create a quad_storage bind group using the pipeline's bgl.
        // In build order 4 this moves into the indirect draw path.
        let quad_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("quad storage bg"),
            layout:  &pipeline.quad_storage_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: vertex_pool.quad_buffer.as_entire_binding(),
            }],
        });
        // Store the bind group on the pool for the renderer to access.
        vertex_pool.bind_group = quad_bg;

        let world = Self::load_world(&store);
        let mut mgr = WorldManager {
            world,
            vertex_pool,
            chunk_draws: std::collections::HashMap::new(),
            store,
            place_voxel: VoxelId::STONE,
            reach: 50.0,
            brush_radius: 0,
        };
        mgr.mesh_all_chunks(device, pipeline);
        log::info!("WorldManager: {} chunks meshed", mgr.chunk_draws.len());
        mgr
    }

    // ── World loading ─────────────────────────────────────────────────────────

    fn load_world(store: &ChunkStore) -> World {
        let params = TerrainParams::default();
        let mut world = World::new();
        let (mut from_disk, mut generated) = (0usize, 0usize);

        for cy in -2i32..=1 {
            for cz in -2i32..=2 {
                for cx in -2i32..=2 {
                    let pos = IVec3::new(cx, cy, cz);
                    match store.load_chunk(pos) {
                        Ok(Some(chunk)) => {
                            log::debug!("persistence: loaded {:?} from disk", pos);
                            world.insert_chunk(pos, chunk);
                            from_disk += 1;
                        }
                        Ok(None) => {
                            world.insert_chunk(pos, generate_chunk(pos, &params));
                            generated += 1;
                        }
                        Err(e) => {
                            log::warn!("persistence: load {:?} failed: {e}", pos);
                            world.insert_chunk(pos, generate_chunk(pos, &params));
                            generated += 1;
                        }
                    }
                }
            }
        }
        log::info!("persistence: {} from disk, {} generated", from_disk, generated);
        world
    }

    // ── Meshing ───────────────────────────────────────────────────────────────

    fn mesh_all_chunks(&mut self, device: &wgpu::Device, pipeline: &OptimisedPipeline) {
        let positions: Vec<IVec3> = self.world.chunks.keys().copied().collect();
        // Need a temporary queue-like mechanism: collect quads first, then upload.
        // We create a dummy queue by using write_buffer through a real queue;
        // but WorldManager::new is called before the queue exists here.
        // Solution: accept queue as parameter via mesh_chunk_upload.
        for pos in positions {
            // Just build the draw record without uploading yet — upload happens
            // in the first frame via upload_pending. For now register the origin buf.
            self.register_chunk_origin(device, pipeline, pos);
        }
    }

    /// Creates the origin uniform buffer and bind group for a chunk,
    /// and pre-populates the vertex pool with greedy quads.
    pub fn upload_chunk(
        &mut self,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        pipeline: &OptimisedPipeline,
        pos:      IVec3,
    ) {
        let Some(chunk) = self.world.get_chunk(&pos) else { return; };
        let quads = mesh_chunk(chunk, None);

        if quads.is_empty() {
            self.chunk_draws.remove(&pos);
            self.vertex_pool.remove_chunk(queue, &pos);
            return;
        }

        let record = match self.vertex_pool.upload_chunk(device, queue, pos, &quads) {
            Some(r) => r,
            None    => { log::warn!("vertex pool full — skipping chunk {:?}", pos); return; }
        };

        let vertex_count = record.quad_count * 6;
        let first_quad   = (record.slot_range.first * crate::vertex_pool::QUADS_PER_SLOT) as u32;

        let origin = [
            (pos.x * CHUNK_SIZE_I) as f32,
            (pos.y * CHUNK_SIZE_I) as f32,
            (pos.z * CHUNK_SIZE_I) as f32,
            0.0f32,
        ];
        let uniform = ChunkOriginUniform { origin };
        let origin_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("chunk origin"),
            contents: bytemuck::bytes_of(&uniform),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let origin_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("chunk origin bg"),
            layout:  &pipeline.chunk_origin_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: origin_buf.as_entire_binding(),
            }],
        });

        self.chunk_draws.insert(pos, ChunkDraw {
            origin_buf, origin_bind_group,
            vertex_count, first_quad,
        });
    }

    /// Creates only the origin buffer (no quad upload) for initial setup.
    fn register_chunk_origin(
        &mut self,
        device:   &wgpu::Device,
        pipeline: &OptimisedPipeline,
        pos:      IVec3,
    ) {
        let origin = [
            (pos.x * CHUNK_SIZE_I) as f32,
            (pos.y * CHUNK_SIZE_I) as f32,
            (pos.z * CHUNK_SIZE_I) as f32,
            0.0f32,
        ];
        let uniform = ChunkOriginUniform { origin };
        let origin_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("chunk origin"),
            contents: bytemuck::bytes_of(&uniform),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let origin_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("chunk origin bg"),
            layout:  &pipeline.chunk_origin_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: origin_buf.as_entire_binding(),
            }],
        });
        // vertex_count = 0 until upload_chunk is called
        self.chunk_draws.insert(pos, ChunkDraw {
            origin_buf, origin_bind_group,
            vertex_count: 0, first_quad: 0,
        });
    }

    /// Uploads all chunks that haven't been uploaded yet (vertex_count == 0).
    /// Called once after the queue is available.
    pub fn flush_pending_uploads(
        &mut self,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        pipeline: &OptimisedPipeline,
    ) {
        let pending: Vec<IVec3> = self.chunk_draws.iter()
            .filter(|(_, d)| d.vertex_count == 0)
            .map(|(pos, _)| *pos)
            .collect();

        log::info!("flush_pending_uploads: uploading {} chunks", pending.len());
        for pos in pending {
            self.upload_chunk(device, queue, pipeline, pos);
        }

        // Report any chunks still at vertex_count=0 after upload attempt.
        // These indicate upload_chunk returned early (empty mesh or pool full).
        let still_zero: Vec<IVec3> = self.chunk_draws.iter()
            .filter(|(_, d)| d.vertex_count == 0)
            .map(|(pos, _)| *pos)
            .collect();
        if !still_zero.is_empty() {
            log::warn!(
                "{} chunk(s) still have vertex_count=0 after upload: {:?}",
                still_zero.len(), still_zero
            );
        }

        log::info!(
            "vertex pool: {} chunks, {} quads total, {} slots used of {}",
            self.vertex_pool.chunk_count(),
            self.vertex_pool.total_quads(),
            self.vertex_pool.allocator.allocated_count,
            crate::vertex_pool::MAX_SLOTS,
        );
    }

    // ── Remeshing ─────────────────────────────────────────────────────────────

    fn remesh_modified(
        &mut self,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        pipeline: &OptimisedPipeline,
        modified: &[IVec3],
    ) {
        let mut to_remesh = HashSet::new();
        let offsets = [
            IVec3::ZERO,
            IVec3::new( 1,0,0), IVec3::new(-1,0,0),
            IVec3::new(0, 1,0), IVec3::new(0,-1,0),
            IVec3::new(0,0, 1), IVec3::new(0,0,-1),
        ];
        for &p in modified {
            for &off in &offsets {
                let cp = chunk_pos_of(p + off);
                if self.world.get_chunk(&cp).is_some() {
                    to_remesh.insert(cp);
                }
            }
        }
        for cp in to_remesh {
            self.upload_chunk(device, queue, pipeline, cp);
        }
        log::debug!(
            "remesh: {} draws, {} dirty chunks pending save",
            self.chunk_draws.len(),
            self.world.dirty_chunks().len()
        );
    }

    // ── Brush ─────────────────────────────────────────────────────────────────

    pub fn raycast(&self, camera: &Camera) -> Option<RayHit> {
        let result = raycast(&self.world, camera.position, camera.forward, self.reach);
        if let Some(ref hit) = result {
            log::info!("raycast HIT: {:?} dist={:.2}", hit.voxel_pos, hit.distance);
        }
        result
    }

    pub fn dig(
        &mut self, device: &wgpu::Device, queue: &wgpu::Queue,
        pipeline: &OptimisedPipeline, hit: &RayHit,
    ) {
        log::info!("dig: {:?} radius={}", hit.voxel_pos, self.brush_radius);
        let modified = remove(&mut self.world, hit, self.brush_radius);
        if !modified.is_empty() {
            self.remesh_modified(device, queue, pipeline, &modified);
        }
    }

    pub fn place(
        &mut self, device: &wgpu::Device, queue: &wgpu::Queue,
        pipeline: &OptimisedPipeline, hit: &RayHit,
    ) {
        log::info!("place: {:?} voxel={:?}", hit.prev_pos, self.place_voxel);
        let modified = place(&mut self.world, hit, self.place_voxel, self.brush_radius);
        if !modified.is_empty() {
            self.remesh_modified(device, queue, pipeline, &modified);
        }
    }

    pub fn cycle_place_voxel(&mut self) {
        self.place_voxel = match self.place_voxel {
            VoxelId::STONE => VoxelId::DIRT,
            VoxelId::DIRT  => VoxelId::GRASS,
            VoxelId::GRASS => VoxelId::SAND,
            VoxelId::SAND  => VoxelId::STONE,
            _              => VoxelId::STONE,
        };
        log::info!("place voxel: {:?}", self.place_voxel);
    }

    pub fn increase_brush(&mut self) {
        self.brush_radius = (self.brush_radius + 1).min(20);
        log::info!("brush radius: {}", self.brush_radius);
    }

    pub fn decrease_brush(&mut self) {
        self.brush_radius = self.brush_radius.saturating_sub(1);
        log::info!("brush radius: {}", self.brush_radius);
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    pub fn save(&mut self) {
        let dirty = self.world.dirty_chunks();
        if dirty.is_empty() {
            log::info!("save: nothing to save");
            return;
        }
        log::info!("save: flushing {} dirty chunk(s)", dirty.len());
        match self.store.flush_dirty(&mut self.world) {
            Ok(n)  => log::info!("save: wrote {n} chunk(s)"),
            Err(e) => log::error!("save: failed: {e}"),
        }
    }

    pub fn dirty_count(&self) -> usize {
        self.world.dirty_chunks().len()
    }
}