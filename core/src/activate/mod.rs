/*
    Appellation: activate <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! This module implements various activation functions for neural networks.
//!
//! ## Traits
//!
//! - [Heavyside]
//! - [LinearActivation]
//! - [Sigmoid]
//! - [Softmax]
//! - [ReLU]
//! - [Tanh]
//!
#[doc(inline)]
pub use self::prelude::*;

pub(crate) mod traits;
pub(crate) mod utils;

mod impls {
    mod impl_binary;
    mod impl_linear;
    mod impl_nonlinear;
}

pub(crate) mod prelude {
    pub use super::traits::*;
    pub use super::utils::*;
}
