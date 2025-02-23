/*
    Appellation: signal <module>
    Contrib: @FL03
*/
//! # Signal Processing
//!
//! This module contains functions for signal processing such as convolution, filtering, and Fourier transforms.

pub mod fourier;

pub(crate) mod prelude {
    pub use super::fourier::prelude::*;
}
