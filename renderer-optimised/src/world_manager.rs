/// WorldManager 
use std::collections::HashSet;

use glam::IVec3;

use voxel_core::{
    gen::{TerrainParams, generate_chunk, mesh_chunk},
    persistence::ChunkStore,
    world::{VoxelId, World, chunk_pos_of},
    camera::Camera,
    input::{RayHit, place, raycast, remove},
};

use crate::indirect::IndirectBuffer;
use crate::pipeline::OptimisedPipeline;
use crate::vertex_pool::VertexPool;


pub struct WorldExtent {
    pub draw_radius:     i32,
    pub vertical_layers: i32,
    pub terrain:         TerrainParams,
}

impl Default for WorldExtent {
    fn default() -> Self {
        Self {
            draw_radius:     8,
            vertical_layers: 4,
            terrain:         TerrainParams::default(),
        }
    }
}


pub struct WorldManager {
    pub world:           World,
    pub vertex_pool:     VertexPool,
    pub indirect_buffer: IndirectBuffer,

    store:        ChunkStore,

    pub place_voxel:   VoxelId,
    pub reach:         f32,
    pub brush_radius:  u32,
}

impl WorldManager {
    const SAVE_PATH: &'static str = "world_optimised.db";

    pub fn new(device: &wgpu::Device, pipeline: &OptimisedPipeline, extent: WorldExtent) -> Self {
        let store = match ChunkStore::open(Self::SAVE_PATH) {
            Ok(s)  => { log::info!("persistence: opened '{}'", Self::SAVE_PATH); s }
            Err(e) => {
                log::warn!("persistence: could not open store ({e}) — edits not saved");
                ChunkStore::open(":memory:").expect("in-memory fallback")
            }
        };

        let mut vertex_pool = VertexPool::new(device);

        let quad_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("quad storage bg"),
            layout:  &pipeline.quad_storage_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: vertex_pool.quad_buffer.as_entire_binding(),
            }],
        });
        vertex_pool.bind_group = quad_bg;

        let world = Self::load_world(&store, &extent);
        let indirect_buffer = IndirectBuffer::new(device, crate::vertex_pool::MAX_SLOTS as u32);

        let mut mgr = WorldManager {
            world,
            vertex_pool,
            indirect_buffer,
            store,
            place_voxel: VoxelId::STONE,
            reach: 50.0,
            brush_radius: 0,
        };
        mgr.mesh_all_chunks(device, pipeline);
        log::info!("WorldManager: {} chunks in pool", mgr.vertex_pool.chunk_count());
        mgr
    }


    fn load_world(store: &ChunkStore, extent: &WorldExtent) -> World {
        // Use the scene's actual terrain params, not the hardcoded default.
        let params = &extent.terrain;
        let mut world = World::new();
        let (mut from_disk, mut generated) = (0usize, 0usize);

        let r = extent.draw_radius;

        // Anchor the vertical range so that sea_level always falls inside it.
        // sea_level is in voxels; convert to chunk Y: sea_chunk = sea_level / 32.
        // Centre the vertical window on sea_chunk so surface chunks are always loaded.
        let sea_chunk  = (params.sea_level as i32).div_euclid(32);
        let half_v     = extent.vertical_layers / 2;
        let cy_min     = sea_chunk - half_v;
        let cy_max     = sea_chunk + half_v - 1;

        log::info!(
            "load_world: draw_radius={} vertical_layers={} sea_chunk={} cy={}..={} → {}×{}×{} = {} chunks",
            r, extent.vertical_layers, sea_chunk, cy_min, cy_max,
            r * 2 + 1, extent.vertical_layers, r * 2 + 1,
            (r * 2 + 1).pow(2) * extent.vertical_layers,
        );

        for cy in cy_min..=cy_max {
            for cz in -r..=r {
                for cx in -r..=r {
                    let pos = IVec3::new(cx, cy, cz);
                    match store.load_chunk(pos) {
                        Ok(Some(chunk)) => {
                            log::debug!("persistence: loaded {:?} from disk", pos);
                            world.insert_chunk(pos, chunk);
                            from_disk += 1;
                        }
                        Ok(None) => {
                            world.insert_chunk(pos, generate_chunk(pos, params));
                            generated += 1;
                        }
                        Err(e) => {
                            log::warn!("persistence: load {:?} failed: {e}", pos);
                            world.insert_chunk(pos, generate_chunk(pos, params));
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

    fn mesh_all_chunks(&mut self, _device: &wgpu::Device, _pipeline: &OptimisedPipeline) {
        // No-op in build order 4 — actual upload happens in flush_pending_uploads
        // once the queue is available. The world is already populated.
    }

    /// Meshes a chunk and uploads its quads into the vertex pool.
    pub fn upload_chunk(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        _pipeline: &OptimisedPipeline,
        pos:    IVec3,
    ) {
        let Some(chunk) = self.world.get_chunk(&pos) else { return; };
        let quads = mesh_chunk(chunk, None);

        if quads.is_empty() {
            self.vertex_pool.remove_chunk(queue, &pos);
            return;
        }

        if self.vertex_pool.upload_chunk(device, queue, pos, &quads).is_none() {
            log::error!(
                "vertex pool full — skipping chunk {:?} \
                 ({} slots used of {}, {} chunks, {} quads total). \
                 Increase MAX_SLOTS in vertex_pool.rs.",
                pos,
                self.vertex_pool.allocator.allocated_count,
                crate::vertex_pool::MAX_SLOTS,
                self.vertex_pool.chunk_count(),
                self.vertex_pool.total_quads(),
            );
        }
    }

    /// Uploads all world chunks into the vertex pool.
    /// Called once on the first frame when the queue is available.
    pub fn flush_pending_uploads(
        &mut self,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        pipeline: &OptimisedPipeline,
    ) {
        let positions: Vec<IVec3> = self.world.chunks.keys().copied().collect();
        log::info!("flush_pending_uploads: uploading {} chunks", positions.len());

        for pos in positions {
            self.upload_chunk(device, queue, pipeline, pos);
        }

        log::info!(
            "vertex pool: {} chunks, {} quads total, {} slots used of {}",
            self.vertex_pool.chunk_count(),
            self.vertex_pool.total_quads(),
            self.vertex_pool.allocator.allocated_count,
            crate::vertex_pool::MAX_SLOTS,
        );

        self.rebuild_indirect(queue);
    }

    /// Rebuilds the compact indirect draw buffer from current pool state.
    pub fn rebuild_indirect(&mut self, queue: &wgpu::Queue) {
        let entries: Vec<(IVec3, crate::vertex_pool::DrawIndirectArgs)> =
            self.vertex_pool.active_draw_args().collect();
        self.indirect_buffer.rebuild(queue, &entries);
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
        self.rebuild_indirect(queue);
        log::debug!(
            "remesh: {} draws, {} dirty chunks pending save",
            self.vertex_pool.chunk_count(),
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