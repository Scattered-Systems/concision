/*
    Appellation: fourier <module>
    Contrib: @FL03
*/
//! # Fast Fourier Transform
//!
//! The `fft` module provides an implementation of the Fast Fourier Transform (FFT) algorithm.
//! The Fast Fourier Transform is an efficient algorithm for computing the Discrete Fourier Transform (DFT).
#[doc(inline)]
pub use self::prelude::*;

mod mode;
pub(crate) mod plan;
pub(crate) mod utils;

pub(crate) mod prelude {
    pub use super::DFT;
    pub use super::mode::*;
    pub use super::plan::*;
    pub use super::utils::*;
}

/// Trait for computing the Discrete Fourier Transform (DFT) of a sequence.
pub trait DFT<T> {
    type Output;

    fn dft(&self) -> Self::Output;
}

#[cfg(test)]
mod tests {
    use super::FftPlan;
    use super::utils::fft_permutation;

    #[test]
    fn test_plan() {
        let samples = 16;

        let plan = FftPlan::new(samples).build();
        assert_eq!(plan.plan(), fft_permutation(16).as_slice());
    }
}
