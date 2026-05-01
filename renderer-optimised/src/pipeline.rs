/// Optimised render pipeline.
/// No vertex buffer — vertex shader reads quads from the storage buffer via vertex_index.
/// Three bind groups: camera (0), chunk origin (1), quad storage (2).

/// Per-chunk world-space origin uniform.
/// Uploaded once per draw call (build order 3) or read from an instance
/// array by `first_instance` (build order 4, indirect draw).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkOriginUniform {
    /// xyz = world-space chunk origin, w = 0 (padding).
    pub origin: [f32; 4],
}

pub struct OptimisedPipeline {
    pub pipeline:              wgpu::RenderPipeline,
    /// Bind group layout for group 1: chunk origin uniform.
    pub chunk_origin_bgl:      wgpu::BindGroupLayout,
    /// Bind group layout for group 2: quad storage buffer.
    /// Shared with the vertex pool's bind group layout.
    pub quad_storage_bgl:      wgpu::BindGroupLayout,
}

impl OptimisedPipeline {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(
        device:                   &wgpu::Device,
        surface_format:           wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Load both shaders as a single concatenated WGSL module.
        // VertexOutput is defined in vert.wgsl and referenced in frag.wgsl.
        let shader_src = concat!(
            include_str!("../shaders/vert.wgsl"),
            include_str!("../shaders/frag.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("optimised shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // ── Group 1: chunk origins storage array ──────────────────────────────
        // One vec4<f32> per pool slot — indexed by first_instance/QUADS_PER_SLOT.
        let chunk_origin_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   Some("optimised chunk origins bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // ── Group 2: quad storage buffer ─────────────────────────────────────
        let quad_storage_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   Some("optimised quad storage bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("optimised pipeline layout"),
                bind_group_layouts: &[
                    camera_bind_group_layout,  // group 0
                    &chunk_origin_bgl,          // group 1
                    &quad_storage_bgl,          // group 2
                ],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("optimised pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:       &shader,
                entry_point:  "vs_main",
                // No vertex buffer — data comes from the storage buffer.
                buffers:      &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:       &shader,
                entry_point:  "fs_main",
                targets:      &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:  Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              Self::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare:       wgpu::CompareFunction::Less,
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview:   None,
            cache:       None,
        });

        OptimisedPipeline { pipeline, chunk_origin_bgl, quad_storage_bgl }
    }
}