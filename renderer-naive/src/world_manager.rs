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
    /// The live voxel world.
    pub world: World,

    /// Meshes uploaded to the GPU — one entry per non-empty chunk.
    pub chunk_draws: HashMap<IVec3, ChunkDraw>,

    /// On-disk persistence store.
    store: ChunkStore,

    /// Voxel type placed by RMB. Cycle with Tab.
    pub place_voxel: VoxelId,

    /// Raycast reach in world units.
    pub reach: f32,

    /// Spherical brush radius (0 = single voxel).
    pub brush_radius: u32,
}

impl WorldManager {
    const SAVE_PATH: &'static str = "world.db";

    /// Creates the manager: opens the store, loads/generates the world,
    /// and uploads initial meshes to the GPU.
    pub fn new(device: &wgpu::Device, pipeline: &NaivePipeline) -> Self {
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

        let world = Self::load_world(&store);
        let chunk_draws = Self::mesh_all_chunks(device, pipeline, &world);
        log::info!("WorldManager: {} draw calls ready", chunk_draws.len());

        WorldManager {
            world,
            chunk_draws,
            store,
            place_voxel: VoxelId::STONE,
            reach: 50.0,
            brush_radius: 0,
        }
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
                            log::warn!("persistence: load failed for {:?}: {e} — generating", pos);
                            world.insert_chunk(pos, generate_chunk(pos, &params));
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
        device: &wgpu::Device,
        pipeline: &NaivePipeline,
        world: &World,
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

    /// Meshes one chunk. Returns `None` if the chunk is empty (nothing to draw).
    pub fn mesh_chunk(
        device: &wgpu::Device,
        pipeline: &NaivePipeline,
        world: &World,
        chunk_pos: IVec3,
    ) -> Option<ChunkDraw> {
        let chunk = world.get_chunk(&chunk_pos)?;
        let (verts, idx) = build_chunk_mesh(chunk, chunk_pos, world);
        if verts.is_empty() {
            return None;
        }

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk ibuf"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        let origin = [
            (chunk_pos.x * CHUNK_SIZE_I) as f32,
            (chunk_pos.y * CHUNK_SIZE_I) as f32,
            (chunk_pos.z * CHUNK_SIZE_I) as f32,
            0.0f32,
        ];
        let chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk uniform"),
            contents: bytemuck::bytes_of(&ChunkUniform { origin }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chunk_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk bg"),
            layout: &pipeline.chunk_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
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

    /// Rebuilds GPU buffers for all chunks that contain any of the given
    /// world-space positions, plus their 6 face-adjacent neighbours.
    ///
    /// Does NOT clear dirty flags — that is the store's responsibility.
    fn remesh_modified(&mut self, device: &wgpu::Device, pipeline: &NaivePipeline, modified: &[IVec3]) {
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
                Some(draw) => {
                    log::debug!("  chunk {:?} -> {} indices", cp, draw.index_count);
                    self.chunk_draws.insert(cp, draw);
                }
                None => {
                    log::debug!("  chunk {:?} -> empty, removed", cp);
                    self.chunk_draws.remove(&cp);
                }
            }
        }

        log::debug!(
            "remesh_modified: done — {} draws, {} dirty pending save",
            self.chunk_draws.len(),
            self.world.dirty_chunks().len(),
        );
    }

    // ── Brush ─────────────────────────────────────────────────────────────────

    /// Casts a ray from the camera into the world.
    pub fn raycast(&self, camera: &Camera) -> Option<RayHit> {
        log::debug!(
            "raycast: origin={:.2?} forward={:.2?} reach={}",
            camera.position, camera.forward, self.reach
        );
        let result = raycast(&self.world, camera.position, camera.forward, self.reach);
        match &result {
            Some(hit) => log::info!(
                "raycast HIT: voxel={:?} prev={:?} dist={:.2}",
                hit.voxel_pos, hit.prev_pos, hit.distance
            ),
            None => log::debug!("raycast: no hit within reach={}", self.reach),
        }
        result
    }

    /// Removes voxels in a sphere around the hit and remeshes.
    pub fn dig(&mut self, device: &wgpu::Device, pipeline: &NaivePipeline, hit: &RayHit) {
        let voxel = self.world.get_voxel(hit.voxel_pos);
        log::info!("dig: target={:?} voxel={:?} radius={}", hit.voxel_pos, voxel, self.brush_radius);

        let modified = remove(&mut self.world, hit, self.brush_radius);
        if modified.is_empty() {
            log::warn!("dig: no voxels removed (chunk not loaded?)");
        } else {
            log::info!("dig: removed {} voxels", modified.len());
            self.remesh_modified(device, pipeline, &modified);
        }
    }

    /// Places `place_voxel` in a sphere around the hit and remeshes.
    pub fn place(&mut self, device: &wgpu::Device, pipeline: &NaivePipeline, hit: &RayHit) {
        log::info!("place: target={:?} voxel={:?} radius={}", hit.prev_pos, self.place_voxel, self.brush_radius);

        let modified = place(&mut self.world, hit, self.place_voxel, self.brush_radius);
        if modified.is_empty() {
            log::warn!("place: no voxels placed (chunk not loaded or placing AIR?)");
        } else {
            log::info!("place: placed {} voxels", modified.len());
            self.remesh_modified(device, pipeline, &modified);
        }
    }

    /// Cycles the active place voxel: Stone → Dirt → Grass → Sand → Stone.
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

    /// Flushes all dirty chunks to disk. Called on F5 and window close.
    pub fn save(&mut self) {
        let dirty = self.world.dirty_chunks();
        if dirty.is_empty() {
            log::info!("save: nothing to save (no dirty chunks)");
            return;
        }

        log::info!("save: flushing {} dirty chunk(s): {:?}", dirty.len(), dirty);

        match self.store.flush_dirty(&mut self.world) {
            Ok(n) => {
                let remaining = self.world.dirty_chunks().len();
                log::info!(
                    "save: wrote {n} chunk(s) to '{}' — {} dirty remaining",
                    Self::SAVE_PATH, remaining
                );
                if remaining > 0 {
                    log::warn!("save: {} chunk(s) still dirty after flush", remaining);
                }
            }
            Err(e) => log::error!("save: flush_dirty failed: {e}"),
        }
    }

    /// How many chunks have unsaved edits.
    pub fn dirty_count(&self) -> usize {
        self.world.dirty_chunks().len()
    }
}