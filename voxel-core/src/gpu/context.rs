use wgpu::{Adapter, Device, DeviceDescriptor, Features, Instance, Limits, Queue};

/// The shared wgpu context — everything both renderers need before they can
/// build pipelines or upload data. Surface creation is handled separately in
/// each renderer because the naive and optimised renderers may configure
/// their swap chains differently.
pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    /// Initialises the wgpu instance, selects an adapter, and creates a
    /// device + queue. This is the blocking (pollster) version used by both
    /// renderers at startup.
    ///
    /// `required_features` lets each renderer request what it needs:
    /// - naive: `Features::empty()`
    /// - optimised: `Features::INDIRECT_FIRST_INSTANCE | Features::MULTI_DRAW_INDIRECT`
    ///   (and optionally `Features::TIMESTAMP_QUERY` when the `gpu-timing` feature is on)
    pub fn new(required_features: Features) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(required_features))
    }

    async fn new_async(required_features: Features) -> Result<Self, GpuError> {
        // wgpu 22: Instance::new takes a value, not a reference.
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // request_adapter is still async in wgpu 22.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        // Features does not implement Sub — use .difference() instead.
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
    /// No suitable adapter found (no GPU, or headless environment without Vulkan/DX12).
    NoAdapter,
    /// The adapter exists but doesn't support required features.
    MissingFeatures(Features),
    /// Device creation failed.
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

    /// This test requires a real GPU. It is skipped gracefully when running
    /// headless (CI, sandbox) by treating NoAdapter as a skip rather than a
    /// failure. It validates that the init path compiles and runs end-to-end.
    #[test]
    fn gpu_context_init_or_skip() {
        match GpuContext::new(Features::empty()) {
            Ok(ctx) => {
                let info = ctx.adapter_info();
                assert!(!info.is_empty(), "adapter info should not be empty");
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