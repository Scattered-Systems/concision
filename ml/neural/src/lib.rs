/*
   Appellation: neural <lib>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-neural
//!
//! This library implements the neural network primitives and specifications.
//!
#![feature(fn_traits)]
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod arch;
pub mod layers;
pub mod loss;
pub mod masks;
pub mod models;
pub mod neurons;
pub mod nn;
pub mod ops;
pub mod params;
pub mod prop;

pub(crate) use concision_core as core;

pub mod prelude {
    pub use crate::arch::*;
    pub use crate::layers::linear::*;
    pub use crate::layers::*;
    pub use crate::loss::*;
    pub use crate::masks::*;
    pub use crate::neurons::activate::*;
    pub use crate::neurons::{Neuron, Node};
    pub use crate::nn::*;
    pub use crate::ops::*;
    pub use crate::params::*;
    pub use crate::prop::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
