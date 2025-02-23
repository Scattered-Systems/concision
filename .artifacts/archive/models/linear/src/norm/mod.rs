/*
    Appellation: norm <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Normalization
//!
//!
pub use self::layer::LayerNorm;

pub mod batch;
pub mod layer;

pub(crate) mod prelude {
    pub use super::batch::prelude::*;
    pub use super::layer::prelude::*;
}
