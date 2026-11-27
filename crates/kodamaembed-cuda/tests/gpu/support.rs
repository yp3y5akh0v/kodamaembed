use kodamaembed_model::Geometry;

pub const GEOMETRIES: [Geometry; 3] = [Geometry::Small, Geometry::Base, Geometry::Large];

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
