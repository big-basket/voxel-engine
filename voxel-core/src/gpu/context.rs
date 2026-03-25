use wgpu::{Adapter, Device, DeviceDescriptor, Features, Instance, Limits, Queue};

pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    /// Creates just the wgpu Instance. Call this before creating the window
    /// surface so the instance exists when `from_surface` is called.
    pub fn create_instance() -> Instance {
        // Prefer Vulkan on Linux — the GLES/EGL backend crashes on Wayland
        // when the surface is created after the instance without a surface hint.
        // On non-Linux platforms, fall back to all backends.
        #[cfg(target_os = "linux")]
        let backends = wgpu::Backends::VULKAN;
        #[cfg(not(target_os = "linux"))]
        let backends = wgpu::Backends::all();

        Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        })
    }

    /// Selects an adapter compatible with `surface`, then creates device + queue.
    /// This is the correct path for Wayland — the adapter is chosen with the
    /// surface in hand so wgpu never needs to switch backends mid-flight.
    pub fn from_surface(
        instance: Instance,
        surface: &wgpu::Surface<'_>,
        required_features: Features,
    ) -> Result<Self, GpuError> {
        pollster::block_on(Self::from_surface_async(instance, surface, required_features))
    }

    async fn from_surface_async(
        instance: Instance,
        surface: &wgpu::Surface<'_>,
        required_features: Features,
    ) -> Result<Self, GpuError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let supported = adapter.features();
        let missing = required_features.difference(supported);
        if !missing.is_empty() {
            return Err(GpuError::MissingFeatures(missing));
        }

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("voxel-engine device"),
                    required_features,
                    required_limits: Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(GpuError::DeviceRequest)?;

        Ok(GpuContext { instance, adapter, device, queue })
    }

    /// Headless path — used by unit tests and non-Wayland contexts where
    /// there is no surface to hint the adapter selection.
    pub fn new_headless(required_features: Features) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_headless_async(required_features))
    }

    async fn new_headless_async(required_features: Features) -> Result<Self, GpuError> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let supported = adapter.features();
        let missing = required_features.difference(supported);
        if !missing.is_empty() {
            return Err(GpuError::MissingFeatures(missing));
        }

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("voxel-engine device"),
                    required_features,
                    required_limits: Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(GpuError::DeviceRequest)?;

        Ok(GpuContext { instance, adapter, device, queue })
    }

    /// Returns a human-readable description of the selected adapter.
    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    MissingFeatures(Features),
    DeviceRequest(wgpu::RequestDeviceError),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no suitable GPU adapter found"),
            GpuError::MissingFeatures(f2) => write!(f, "adapter missing features: {f2:?}"),
            GpuError::DeviceRequest(e) => write!(f, "device request failed: {e}"),
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
                let info = ctx.adapter_info();
                assert!(!info.is_empty());
                println!("GPU adapter: {info}");
            }
            Err(GpuError::NoAdapter) => {
                println!("no GPU adapter available, skipping");
            }
            Err(e) => panic!("unexpected GPU init error: {e}"),
        }
    }

    #[test]
    fn missing_features_error_is_descriptive() {
        let err = GpuError::MissingFeatures(Features::TIMESTAMP_QUERY);
        let msg = err.to_string();
        assert!(msg.contains("missing"), "error message: {msg}");
    }
}