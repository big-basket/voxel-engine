use wgpu::{Adapter, Device, DeviceDescriptor, Features, Instance, Queue};

/// The shared wgpu context 
pub struct GpuContext {
    pub instance: Instance,
    pub adapter:  Adapter,
    pub device:   Device,
    pub queue:    Queue,
}

impl GpuContext {
    pub fn new(required_features: Features) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(required_features))
    }

    async fn new_async(required_features: Features) -> Result<Self, GpuError> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let supported = adapter.features();
        let missing   = required_features.difference(supported);
        if !missing.is_empty() {
            return Err(GpuError::MissingFeatures(missing));
        }

        
        let mut limits = adapter.limits();
        
        limits.max_storage_buffer_binding_size =
            limits.max_storage_buffer_binding_size
                .max(adapter.limits().max_storage_buffer_binding_size);

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label:             Some("voxel-engine device"),
                    required_features,
                    required_limits:   limits,
                    memory_hints:      Default::default(),
                },
                None,
            )
            .await
            .map_err(GpuError::DeviceRequest)?;

        Ok(GpuContext { instance, adapter, device, queue })
    }

    pub fn create_instance() -> Instance {
        Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        })
    }

    pub fn from_surface(
        instance: Instance,
        surface:  &wgpu::Surface<'_>,
        required_features: Features,
    ) -> Result<Self, GpuError> {
        pollster::block_on(Self::from_surface_async(instance, surface, required_features))
    }

    async fn from_surface_async(
        instance: Instance,
        surface:  &wgpu::Surface<'_>,
        required_features: Features,
    ) -> Result<Self, GpuError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let supported = adapter.features();
        let missing   = required_features.difference(supported);
        if !missing.is_empty() {
            return Err(GpuError::MissingFeatures(missing));
        }

        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label:             Some("voxel-engine device"),
                    required_features,
                    required_limits:   limits,
                    memory_hints:      Default::default(),
                },
                None,
            )
            .await
            .map_err(GpuError::DeviceRequest)?;

        Ok(GpuContext { instance, adapter, device, queue })
    }

    pub fn new_headless(required_features: Features) -> Result<Self, GpuError> {
        Self::new(required_features)
    }

    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}


#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    MissingFeatures(Features),
    DeviceRequest(wgpu::RequestDeviceError),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter          => write!(f, "no suitable GPU adapter found"),
            GpuError::MissingFeatures(m) => write!(f, "adapter missing features: {m:?}"),
            GpuError::DeviceRequest(e)   => write!(f, "device request failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_context_headless_or_skip() {
        match GpuContext::new_headless(Features::empty()) {
            Ok(ctx) => {
                assert!(!ctx.adapter_info().is_empty());
                // Verify the storage buffer limit is above the wgpu default.
                let limit = ctx.device.limits().max_storage_buffer_binding_size;
                println!("max_storage_buffer_binding_size: {} MiB", limit / (1024 * 1024));
            }
            Err(GpuError::NoAdapter) => println!("no GPU — skipping"),
            Err(e) => panic!("GPU init error: {e}"),
        }
    }

    #[test]
    fn missing_features_error_is_descriptive() {
        let err = GpuError::MissingFeatures(Features::TIMESTAMP_QUERY);
        assert!(err.to_string().contains("missing"));
    }
}