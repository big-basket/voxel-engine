use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::IVec3;
use wgpu::{
    CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, SurfaceConfiguration, TextureUsages, TextureViewDescriptor,
    util::DeviceExt,
};
use winit::window::Window;

use voxel_core::{
    camera::{Camera, CameraUniform},
    gen::{TerrainParams, generate_chunk},
    gpu::{GpuContext, GpuError, create_uniform_buffer, write_uniform},
    input::{RayHit, place, raycast, remove},
    world::{VoxelId, World, CHUNK_SIZE_I, chunk_pos_of},
};

use crate::mesh::build_chunk_mesh;
use crate::pipeline::{ChunkUniform, NaivePipeline};

// ── Per-chunk GPU resources ───────────────────────────────────────────────────

#[allow(dead_code)]
struct ChunkDraw {
    chunk_pos:   IVec3,
    vertex_buf:  wgpu::Buffer,
    index_buf:   wgpu::Buffer,
    index_count: u32,
    chunk_buf:   wgpu::Buffer,
    chunk_bind_group: wgpu::BindGroup,
}

// ── Renderer ──────────────────────────────────────────────────────────────────

pub struct NaiveRenderer {
    pub window: Arc<Window>,

    pub gpu: GpuContext,
    surface: wgpu::Surface<'static>,
    config: SurfaceConfiguration,

    pub camera_buf: wgpu::Buffer,
    #[allow(dead_code)]
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_bind_group: wgpu::BindGroup,

    depth_texture: wgpu::Texture,
    depth_view:    wgpu::TextureView,

    pipeline:    NaivePipeline,
    chunk_draws: HashMap<IVec3, ChunkDraw>,

    /// The live world — kept here so the brush can mutate it each frame.
    pub world: World,

    /// Voxel type placed by right-click. Cycle with Tab.
    pub place_voxel: VoxelId,

    /// Raycast reach in world units.
    pub reach: f32,

    /// Radius of the brush for digging and placing.
    pub brush_radius: u32,
}

impl NaiveRenderer {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(window: Arc<Window>, width: u32, height: u32) -> Result<Self, GpuError> {
        let instance = GpuContext::create_instance();
        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("create surface");
        let gpu = GpuContext::from_surface(instance, &surface, wgpu::Features::empty())?;
        log::info!("GPU: {}", gpu.adapter_info());

        let caps = surface.get_capabilities(&gpu.adapter);
        let surface_format = caps.formats.iter()
            .find(|f| f.is_srgb()).copied()
            .unwrap_or(caps.formats[0]);
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::AutoVsync
        };
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &config);

        let camera_buf = create_uniform_buffer::<CameraUniform>(&gpu.device, "camera uniform");

        let camera_bind_group_layout =
            gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(CameraUniform::SIZE),
                    },
                    count: None,
                }],
            });

        let camera_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bg"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let (depth_texture, depth_view) =
            Self::create_depth_texture(&gpu.device, width.max(1), height.max(1));

        let pipeline = NaivePipeline::new(
            &gpu.device, surface_format,
            &camera_bind_group_layout, Self::DEPTH_FORMAT,
        );

        // Build world and initial meshes
        let world = Self::build_world();
        let chunk_draws = Self::mesh_all_chunks(&gpu.device, &pipeline, &world);
        log::info!("Generated {} chunk draw calls", chunk_draws.len());

        Ok(NaiveRenderer {
            window, gpu, surface, config,
            camera_buf, camera_bind_group_layout, camera_bind_group,
            depth_texture, depth_view,
            pipeline, chunk_draws, world,
            place_voxel: VoxelId::STONE,
            reach: 50.0,
            brush_radius: 0,
        })
    }

    // ── World construction ────────────────────────────────────────────────────

    fn build_world() -> World {
        let params = TerrainParams::default();
        let mut world = World::new();
        for cy in -2i32..=1 {
            for cz in -2i32..=2 {
                for cx in -2i32..=2 {
                    let pos = IVec3::new(cx, cy, cz);
                    world.insert_chunk(pos, generate_chunk(pos, &params));
                }
            }
        }
        world
    }

    fn mesh_all_chunks(
        device: &wgpu::Device,
        pipeline: &NaivePipeline,
        world: &World,
    ) -> HashMap<IVec3, ChunkDraw> {
        let positions: Vec<IVec3> = world.chunks.keys().copied().collect();
        let mut draws = HashMap::new();
        for chunk_pos in positions {
            if let Some(draw) = Self::mesh_chunk(device, pipeline, world, chunk_pos) {
                draws.insert(chunk_pos, draw);
            }
        }
        draws
    }

    /// Meshes a single chunk. Returns None if the mesh is empty (all-air chunk).
    fn mesh_chunk(
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

    // ── Brush ─────────────────────────────────────────────────────────────────

    /// Casts a ray from the camera and returns the hit, if any.
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

    /// Removes voxels at the hit position and remeshes affected chunks.
    pub fn dig(&mut self, hit: &RayHit) {
        let voxel = self.world.get_voxel(hit.voxel_pos);
        log::info!("dig: target={:?} voxel={:?} radius={}", hit.voxel_pos, voxel, self.brush_radius);
        
        let modified = remove(&mut self.world, hit, self.brush_radius);
        if !modified.is_empty() {
            log::info!("dig: removed {} voxels, remeshing...", modified.len());
            self.remesh_modified(&modified);
        } else {
            log::warn!("dig: no voxels removed (none loaded?)");
        }
    }

    /// Places `self.place_voxel` at the face in front of the hit and remeshes.
    pub fn place(&mut self, hit: &RayHit) {
        log::info!(
            "place: target={:?} voxel={:?} radius={}",
            hit.prev_pos, self.place_voxel, self.brush_radius
        );
        
        let modified = place(&mut self.world, hit, self.place_voxel, self.brush_radius);
        if !modified.is_empty() {
            log::info!("place: placed {} voxels, remeshing...", modified.len());
            self.remesh_modified(&modified);
        } else {
            log::warn!("place: no voxels placed (none loaded or AIR?)");
        }
    }

    /// Cycles through placeable voxel types.
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
        log::info!("increase_brush: radius is now {}", self.brush_radius);
    }

    pub fn decrease_brush(&mut self) {
        self.brush_radius = self.brush_radius.saturating_sub(1);
        log::info!("decrease_brush: radius is now {}", self.brush_radius);
    }

    /// Remeshes chunks containing the modified voxels and their face-adjacent neighbours.
    fn remesh_modified(&mut self, modified: &[IVec3]) {
        let mut to_remesh = HashSet::new();

        let neighbours = [
            IVec3::ZERO,
            IVec3::new( 1, 0, 0), IVec3::new(-1, 0, 0),
            IVec3::new( 0, 1, 0), IVec3::new( 0,-1, 0),
            IVec3::new( 0, 0, 1), IVec3::new( 0, 0,-1),
        ];

        for &pos in modified {
            for &offset in &neighbours {
                let nb_pos = pos + offset;
                let chunk_pos = chunk_pos_of(nb_pos);
                if self.world.get_chunk(&chunk_pos).is_some() {
                    to_remesh.insert(chunk_pos);
                }
            }
        }

        log::debug!("remesh_modified: remeshing {} chunk(s)", to_remesh.len());

        for chunk_pos in to_remesh {
            if let Some(chunk) = self.world.get_chunk_mut(&chunk_pos) {
                chunk.mark_clean();
            }
            match Self::mesh_chunk(&self.gpu.device, &self.pipeline, &self.world, chunk_pos) {
                Some(draw) => {
                    log::debug!("remesh_modified: chunk {:?} -> {} indices", chunk_pos, draw.index_count);
                    self.chunk_draws.insert(chunk_pos, draw);
                }
                None => {
                    log::debug!("remesh_modified: chunk {:?} -> empty (removed from draws)", chunk_pos);
                    self.chunk_draws.remove(&chunk_pos);
                }
            }
        }

        log::info!("remesh_modified: done. total draw calls: {}", self.chunk_draws.len());
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        let (tex, view) = Self::create_depth_texture(&self.gpu.device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
    }

    // ── Render ────────────────────────────────────────────────────────────────

    pub fn render(&mut self, camera: &Camera) -> Result<(), wgpu::SurfaceError> {
        write_uniform(
            &self.gpu.queue,
            &self.camera_buf,
            &CameraUniform::from_camera(camera),
        );

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = self.gpu.device.create_command_encoder(
            &CommandEncoderDescriptor { label: Some("naive frame") },
        );

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("naive pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color { r: 0.53, g: 0.81, b: 0.98, a: 1.0 }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);

            for draw in self.chunk_draws.values() {
                pass.set_bind_group(1, &draw.chunk_bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buf.slice(..));
                pass.set_index_buffer(draw.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..draw.index_count, 0, 0..1);
            }
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    #[allow(dead_code)]
    pub fn surface_format(&self) -> wgpu::TextureFormat { self.config.format }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        (texture, view)
    }
}