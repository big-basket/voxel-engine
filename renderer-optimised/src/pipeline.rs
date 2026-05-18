/// Optimised render pipeline.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkOriginUniform {
    pub origin: [f32; 4],
}

pub struct OptimisedPipeline {
    pub pipeline:              wgpu::RenderPipeline,
    pub chunk_origin_bgl:      wgpu::BindGroupLayout,
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
        let shader_src = concat!(
            include_str!("../shaders/vert.wgsl"),
            include_str!("../shaders/frag.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("optimised shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // chunk origins storage array
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

        // quad storage buffer 
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
                // No vertex buffer 
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