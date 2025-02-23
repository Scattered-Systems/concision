/*
    Appellation: proton-neural <lib>
    Contrib: @FL03
*/
//! This crates provides a set of tools to create and train neural networks.
// #![feature(autodiff)]
#[allow(unused_imports)]
#[doc(inline)]
pub use self::{
    error::*, ops::prelude::*, traits::prelude::*, types::prelude::*, utils::prelude::*,
};

#[doc(inline)]
pub use concision_math as math;

#[macro_use]
pub(crate) mod macros;

pub mod activate;
pub mod error;
#[cfg(feature = "rand")]
pub mod init;
pub mod loss;
pub mod models;
pub mod nn;

pub mod ops {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod adjust;
    pub mod fill;
    pub mod matmul;
    pub mod num;
    pub mod pad;
    pub mod reshape;
    pub mod tensor;

    pub(crate) mod prelude {
        pub use super::adjust::*;
        pub use super::fill::*;
        pub use super::matmul::*;
        pub use super::num::*;
        pub use super::pad::*;
        pub use super::reshape::*;
        pub use super::tensor::*;
    }
}

pub mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod activation;
    pub mod create;
    pub mod model;
    pub mod predict;
    pub mod tensor;
    pub mod train;

    pub(crate) mod prelude {
        pub use super::activation::*;
        pub use super::create::*;
        pub use super::model::*;
        pub use super::predict::*;
        pub use super::tensor::*;
        pub use super::train::*;
    }
}

pub mod types {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod features;
    pub mod layer;
    pub mod params;
    pub mod perceptron;

    pub(crate) mod prelude {
        pub use super::features::*;
        pub use super::layer::*;
        pub use super::params::*;
        pub use super::perceptron::*;
    }
}

pub mod utils {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod checks;
    pub mod tensor;

    pub(crate) mod prelude {
        pub use super::checks::*;
        pub use super::tensor::*;
    }
}

#[allow(unused_imports)]
pub mod prelude {
    pub use concision_math::prelude::*;

    #[cfg(feature = "rand")]
    pub use crate::init::prelude::*;
    pub use crate::loss::prelude::*;
    pub use crate::models::prelude::*;
    pub use crate::ops::prelude::*;
    pub use crate::traits::prelude::*;
    pub use crate::types::prelude::*;
}
