use kodamaembed_model::Geometry;
use kodamaembed_vulkan::{Binding, Descriptors, Device, Kernel, KernelSpec, Push, Recorder};

pub const GEOMETRIES: [Geometry; 3] = [Geometry::Small, Geometry::Base, Geometry::Large];

/// Specialization constant 0 of the row-wise shaders: a quarter of the width.
pub fn quarter(geometry: Geometry) -> [u32; 1] {
    [(geometry.hidden() / 4) as u32]
}

/// Deterministic pseudo-random values (SplitMix64) so failures reproduce.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    pub fn symmetric(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }

    pub fn fill(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.symmetric() * scale).collect()
    }

    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// Cumulative sequence lengths starting at 0.
pub fn cumulative(lengths: &[usize]) -> Vec<i32> {
    std::iter::once(0)
        .chain(lengths.iter().scan(0, |total, &len| {
            *total += i32::try_from(len).unwrap();
            Some(*total)
        }))
        .collect()
}

/// Runs one dispatch of the kernel `spec` describes over `buffers` and waits.
pub fn dispatch(
    device: &Device,
    spec: &KernelSpec,
    buffers: &[Binding],
    push: &Push,
    groups: [u32; 3],
) {
    let kernel = unsafe { Kernel::new(device, spec) }.unwrap();
    let descriptors = Descriptors::new(device, 1, buffers.len() as u32).unwrap();
    let set = unsafe { descriptors.bind(&kernel, buffers) }.unwrap();
    device
        .submit(|command| unsafe {
            Recorder::new(device, command).dispatch(&kernel, set, push, groups);
            Ok(())
        })
        .unwrap();
}

/// Every Vulkan GPU the loader reports, opened.
pub fn devices() -> Vec<Device> {
    let adapters = Device::adapters().unwrap();
    assert!(!adapters.is_empty(), "no Vulkan GPU");
    adapters
        .iter()
        .map(|adapter| Device::new(adapter.index).unwrap())
        .collect()
}

/// Every Vulkan GPU with cooperative matrices, opened.
pub fn matrix_devices() -> Vec<Device> {
    let capable: Vec<Device> = devices()
        .into_iter()
        .filter(|device| device.cooperative_matrix)
        .collect();
    assert!(!capable.is_empty(), "no cooperative-matrix GPU");
    capable
}
