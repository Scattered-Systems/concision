/*
   Appellation: concision-kan <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Kolmogorov-Arnold Networks (KAN)
//!
//! Kolmogorov-Arnold Networks (KAN) are a novel class of neural networks based on the Kolmogorov-Arnold Representation Theorem.
//! KANs propose a fundamental shift in network architecture, elegantly blending splines with Multi-Layer Perceptrons.
//!
//! While scaling these models has yet to be fully explored, KANs have demonstrated their ability to rival traditional neural networks in terms of performance
//! with far fewer parameters.
//! These models have already demonstrated that they are viable alternatives to traditional multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs).
//!
//! ### Resources
//!
//! - [Kolmogorov-Arnold Representation](https://arxiv.org/abs/2404.19756)
//!
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
extern crate concision_core as concision;
extern crate ndarray as nd;

#[macro_use]
pub(crate) mod macros;
pub(crate) mod primitives;

#[doc(inline)]
pub use self::actor::Actor;
#[doc(inline)]
pub use self::model::KAN;

pub mod actor;
pub mod model;

pub mod prelude {
    pub use super::model::prelude::*;
}
