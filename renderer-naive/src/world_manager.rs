/// WorldManager — owns the live world, chunk GPU buffers, brush state,
/// and persistence. The renderer delegates all non-GPU-surface concerns here.
use std::collections::{HashMap, HashSet};

use glam::IVec3;
use wgpu::util::DeviceExt;

use voxel_core::{
    gen::{TerrainParams, generate_chunk},
    input::{RayHit, place, raycast, remove},
    persistence::ChunkStore,
    world::{VoxelId, World, CHUNK_SIZE_I, chunk_pos_of},
    camera::Camera,
};

use crate::mesh::build_chunk_mesh;
use crate::pipeline::{ChunkUniform, NaivePipeline};

// ── WorldExtent ───────────────────────────────────────────────────────────────

/// Spatial extent of the world to load, derived from the scene config.
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

// ── Per-chunk GPU resources ───────────────────────────────────────────────────

/// All GPU-side buffers for one renderable chunk.
#[allow(dead_code)]
pub struct ChunkDraw {
    pub chunk_pos:        IVec3,
    pub vertex_buf:       wgpu::Buffer,
    pub index_buf:        wgpu::Buffer,
    pub index_count:      u32,
    pub chunk_buf:        wgpu::Buffer,
    pub chunk_bind_group: wgpu::BindGroup,
}

// ── WorldManager ──────────────────────────────────────────────────────────────

pub struct WorldManager {
    pub world:       World,
    pub chunk_draws: HashMap<IVec3, ChunkDraw>,
    store:           ChunkStore,
    pub place_voxel: VoxelId,
    pub reach:       f32,
    pub brush_radius: u32,
}

impl WorldManager {
    const SAVE_PATH: &'static str = "world.db";

    pub fn new(device: &wgpu::Device, pipeline: &NaivePipeline, extent: WorldExtent) -> Self {
        let store = match ChunkStore::open(Self::SAVE_PATH) {
            Ok(s) => {
                log::info!("persistence: opened store at '{}'", Self::SAVE_PATH);
                s
            }
            Err(e) => {
                log::warn!("persistence: could not open store ({e}) — edits will not be saved");
                ChunkStore::open(":memory:").expect("in-memory fallback")
            }
        };

        let world       = Self::load_world(&store, &extent);
        let chunk_draws = Self::mesh_all_chunks(device, pipeline, &world);
        log::info!("WorldManager: {} draw calls ready", chunk_draws.len());

        WorldManager {
            world,
            chunk_draws,
            store,
            place_voxel: VoxelId::STONE,
            reach:        50.0,
            brush_radius: 0,
        }
    }

    // ── World loading ─────────────────────────────────────────────────────────

    fn load_world(store: &ChunkStore, extent: &WorldExtent) -> World {
        let params = &extent.terrain;
        let mut world = World::new();
        let (mut from_disk, mut generated) = (0usize, 0usize);

        let r = extent.draw_radius;

        // Anchor the vertical range on the sea-level chunk so surface chunks
        // are always loaded regardless of vertical_layers setting.
        let sea_chunk = (params.sea_level as i32).div_euclid(32);
        let half_v    = extent.vertical_layers / 2;
        let cy_min    = sea_chunk - half_v;
        let cy_max    = sea_chunk + half_v - 1;

        log::info!(
            "load_world: draw_radius={} vertical_layers={} sea_chunk={} cy={}..={} \
             → {}×{}×{} = {} chunks",
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
                            log::warn!("persistence: load failed for {:?}: {e} — generating", pos);
                            world.insert_chunk(pos, generate_chunk(pos, params));
                            generated += 1;
                        }
                    }
                }
            }
        }

        log::info!(
            "persistence: world ready — {} from disk, {} generated ({} total)",
            from_disk, generated, world.chunks.len()
        );
        world
    }

    // ── Chunk meshing ─────────────────────────────────────────────────────────

    fn mesh_all_chunks(
        device:   &wgpu::Device,
        pipeline: &NaivePipeline,
        world:    &World,
    ) -> HashMap<IVec3, ChunkDraw> {
        let positions: Vec<IVec3> = world.chunks.keys().copied().collect();
        let mut draws = HashMap::new();
        for pos in positions {
            if let Some(draw) = Self::mesh_chunk(device, pipeline, world, pos) {
                draws.insert(pos, draw);
            }
        }
        draws
    }

    pub fn mesh_chunk(
        device:    &wgpu::Device,
        pipeline:  &NaivePipeline,
        world:     &World,
        chunk_pos: IVec3,
    ) -> Option<ChunkDraw> {
        let chunk = world.get_chunk(&chunk_pos)?;
        let (verts, idx) = build_chunk_mesh(chunk, chunk_pos, world);
        if verts.is_empty() { return None; }

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("chunk vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("chunk ibuf"),
            contents: bytemuck::cast_slice(&idx),
            usage:    wgpu::BufferUsages::INDEX,
        });
        let origin = [
            (chunk_pos.x * CHUNK_SIZE_I) as f32,
            (chunk_pos.y * CHUNK_SIZE_I) as f32,
            (chunk_pos.z * CHUNK_SIZE_I) as f32,
            0.0f32,
        ];
        let chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("chunk uniform"),
            contents: bytemuck::bytes_of(&ChunkUniform { origin }),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chunk_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("chunk bg"),
            layout:  &pipeline.chunk_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: chunk_buf.as_entire_binding(),
            }],
        });

        Some(ChunkDraw {
            chunk_pos,
            vertex_buf, index_buf,
            index_count: idx.len() as u32,
            chunk_buf, chunk_bind_group,
        })
    }

    fn remesh_modified(
        &mut self,
        device:   &wgpu::Device,
        pipeline: &NaivePipeline,
        modified: &[IVec3],
    ) {
        let mut to_remesh = HashSet::new();
        let face_offsets = [
            IVec3::ZERO,
            IVec3::new( 1, 0, 0), IVec3::new(-1, 0, 0),
            IVec3::new( 0, 1, 0), IVec3::new( 0,-1, 0),
            IVec3::new( 0, 0, 1), IVec3::new( 0, 0,-1),
        ];
        for &pos in modified {
            for &off in &face_offsets {
                let cp = chunk_pos_of(pos + off);
                if self.world.get_chunk(&cp).is_some() {
                    to_remesh.insert(cp);
                }
            }
        }
        log::debug!("remesh_modified: rebuilding {} chunk(s)", to_remesh.len());
        for cp in to_remesh {
            match Self::mesh_chunk(device, pipeline, &self.world, cp) {
                Some(draw) => { self.chunk_draws.insert(cp, draw); }
                None       => { self.chunk_draws.remove(&cp); }
            }
        }
    }

    // ── Brush ─────────────────────────────────────────────────────────────────

    pub fn raycast(&self, camera: &Camera) -> Option<RayHit> {
        let result = raycast(&self.world, camera.position, camera.forward, self.reach);
        if let Some(ref hit) = result {
            log::info!("raycast HIT: voxel={:?} prev={:?} dist={:.2}",
                hit.voxel_pos, hit.prev_pos, hit.distance);
        }
        result
    }

    pub fn dig(&mut self, device: &wgpu::Device, pipeline: &NaivePipeline, hit: &RayHit) {
        log::info!("dig: target={:?} radius={}", hit.voxel_pos, self.brush_radius);
        let modified = remove(&mut self.world, hit, self.brush_radius);
        if !modified.is_empty() {
            self.remesh_modified(device, pipeline, &modified);
        }
    }

    pub fn place(&mut self, device: &wgpu::Device, pipeline: &NaivePipeline, hit: &RayHit) {
        log::info!("place: target={:?} voxel={:?} radius={}", hit.prev_pos, self.place_voxel, self.brush_radius);
        let modified = place(&mut self.world, hit, self.place_voxel, self.brush_radius);
        if !modified.is_empty() {
            self.remesh_modified(device, pipeline, &modified);
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
        log::info!("cycle_place_voxel: now placing {:?}", self.place_voxel);
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
            Ok(n)  => log::info!("save: wrote {n} chunk(s) to '{}'", Self::SAVE_PATH),
            Err(e) => log::error!("save: flush_dirty failed: {e}"),
        }
    }

    pub fn dirty_count(&self) -> usize {
        self.world.dirty_chunks().len()
    }
}