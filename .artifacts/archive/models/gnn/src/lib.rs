/*
   Appellation: concision-gnn <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Graph Neural Networks (GNN)
//!
//! This library implements the framework for building graph-based neural networks.
//!
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_name = "concision_gnn"]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision_core as concision;
extern crate ndarray as nd;

#[doc(inline)]
pub use self::model::*;

pub(crate) mod model;

pub mod params;

pub mod prelude {
    pub use crate::model::prelude::*;
    pub use crate::params::prelude::*;
}
