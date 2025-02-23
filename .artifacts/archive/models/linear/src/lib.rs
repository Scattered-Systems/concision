/*
   Appellation: concision-linear <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Linear Models
//!
//! This library works to provide the necessary tools for creating and training linear models.
//! The primary focus is on the [Linear] model, which is a simple linear model that can be used
//! for regression or classification tasks.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision_core as concision;
extern crate ndarray as nd;
// extern crate ndarray_stats as ndstats;

pub use self::model::{Features, Layout, Linear, LinearConfig};
pub use self::norm::LayerNorm;
pub use self::params::{ParamsBase, mode::*};
#[allow(unused_imports)]
pub use self::{primitives::*, traits::*, utils::*};

#[macro_use]
pub(crate) mod macros;
pub(crate) mod primitives;
#[macro_use]
pub(crate) mod seal;
pub(crate) mod utils;

#[doc(hidden)]
pub mod conv;
#[doc(hidden)]
pub mod dense;
#[doc(hidden)]
pub mod mlp;
pub mod model;
pub mod norm;
pub mod params;
pub mod traits;

mod impls {
    pub mod impl_rand;

    pub mod model {
        pub mod impl_linear;
        pub mod impl_model;
    }

    pub mod params {
        pub mod impl_from;
        pub mod impl_params;
        pub mod impl_serde;
    }
}

pub mod prelude {
    pub use crate::mlp::prelude::*;
    pub use crate::model::prelude::*;
    pub use crate::norm::prelude::*;
    pub use crate::params::prelude::*;
    pub use crate::traits::*;
}
