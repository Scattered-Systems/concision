/*
    Appellation: prop <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Propagation
//!
//! This module describes the propagation of data through a neural network.
pub use self::{kinds::*, propagation::*, utils::*};

pub(crate) mod kinds;
pub(crate) mod propagation;

pub(crate) mod utils {}
