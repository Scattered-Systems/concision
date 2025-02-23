/*
   Appellation: concision-transformers <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Transformers
//!
//! ### Resources
//!
//! - [Attention is All You Need](https://arxiv.org/abs/1706.03762)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision_core as concision;
extern crate concision_linear as linear;
extern crate ndarray as nd;

#[doc(inline)]
pub use self::{
    attention::prelude::*, ops::prelude::*, params::prelude::*, primitives::*,
    transformer::Transformer,
};

#[macro_use]
pub(crate) mod macros;
pub(crate) mod primitives;
pub(crate) mod transformer;

pub mod attention;
pub mod codec;
pub mod config;
pub mod model;
pub mod ops;
pub mod params;

mod impls {
    mod impl_head;
    mod impl_linalg;
    mod impl_params;
}

pub mod prelude {
    pub use super::Transformer;
    pub use super::attention::prelude::*;
    pub use super::params::prelude::*;
}
