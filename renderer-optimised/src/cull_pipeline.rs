/// Frustum cull compute pipeline.

use voxel_core::camera::CameraUniform;

const WORKGROUP_SIZE: u32 = 64;

pub struct CullPipeline {
    pub pipeline: wgpu::ComputePipeline,

    pub camera_bgl: wgpu::BindGroupLayout,

    pub origins_bgl: wgpu::BindGroupLayout,

    pub indirect_bgl: wgpu::BindGroupLayout,
}

impl CullPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("cull shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/cull.wgsl").into()
            ),
        });

        let camera_bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("cull camera bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(CameraUniform::SIZE),
                    },
                    count: None,
                }],
            }
        );

        let origins_bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("cull origins bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }
        );

        let indirect_bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("cull indirect bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("cull pipeline layout"),
            bind_group_layouts:   &[&camera_bgl, &origins_bgl, &indirect_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("cull pipeline"),
            layout:      Some(&layout),
            module:      &shader,
            entry_point: "cs_cull",
            compilation_options: Default::default(),
            cache: None,
        });

        CullPipeline { pipeline, camera_bgl, origins_bgl, indirect_bgl }
    }

    pub fn dispatch_size(draw_count: u32) -> u32 {
        draw_count.div_ceil(WORKGROUP_SIZE)
    }
}