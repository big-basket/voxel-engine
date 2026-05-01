/// Frustum cull compute pipeline.
///
/// Runs once per frame before the render pass. Each thread processes one
/// indirect draw entry, testing its chunk AABB against the 6 camera frustum
/// planes. Culled chunks have their instance_count set to 0 in the indirect
/// buffer — multi_draw_indirect skips them automatically.
///
/// The indirect buffer must be reset to instance_count=1 before each cull pass
/// so that chunks visible last frame but culled this frame are correctly hidden,
/// and chunks culled last frame but visible this frame are correctly shown.

use voxel_core::camera::CameraUniform;

const WORKGROUP_SIZE: u32 = 64;

pub struct CullPipeline {
    pub pipeline: wgpu::ComputePipeline,

    /// @group(0): camera uniform (shared with render pipeline)
    pub camera_bgl: wgpu::BindGroupLayout,

    /// @group(1): chunk origins storage buffer (read-only)
    pub origins_bgl: wgpu::BindGroupLayout,

    /// @group(2): indirect draw buffer (read_write — writes instance_count)
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

        // @group(0) @binding(0): camera uniform
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

        // @group(1) @binding(0): chunk origins (read-only storage)
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

        // @group(2) @binding(0): indirect draw buffer (read_write)
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

    /// Number of workgroups needed to cover `draw_count` threads.
    pub fn dispatch_size(draw_count: u32) -> u32 {
        draw_count.div_ceil(WORKGROUP_SIZE)
    }
}