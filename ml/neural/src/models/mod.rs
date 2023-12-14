/*
    Appellation: models <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
pub use self::{config::*, model::*, modes::*, params::*, utils::*};

pub(crate) mod config;
pub(crate) mod model;
pub(crate) mod modes;
pub(crate) mod params;

pub mod exp;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
