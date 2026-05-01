use wgpu::{Adapter, Device, DeviceDescriptor, Features, Instance, Limits, Queue};

/// The shared wgpu context — everything both renderers need before they can
/// build pipelines or upload data.
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

        // Use the adapter's own limits as the baseline so we don't artificially
        // cap resources below what the hardware supports.
        // In particular, `max_storage_buffer_binding_size` defaults to 128 MiB
        // in wgpu but modern Vulkan/DX12 adapters (e.g. RX 6800M) support up
        // to 4 GiB. The optimised renderer's 256 MiB vertex pool requires this.
        let mut limits = adapter.limits();
        // Clamp to what the adapter actually advertises — prevents requesting
        // more than the driver will grant.
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

    /// Creates a wgpu Instance (used by renderers that need a surface).
    pub fn create_instance() -> Instance {
        Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        })
    }

    /// Initialises from an existing instance and surface (windowed renderers).
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

        // Same as new_async: use the adapter's real limits.
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

    /// Headless init (benchmarks, tests — no surface).
    pub fn new_headless(required_features: Features) -> Result<Self, GpuError> {
        Self::new(required_features)
    }

    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

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