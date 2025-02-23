/*
    Appellation: batch <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Batch Normalization
//!
//!
pub use self::model::*;

mod model;

pub(crate) mod prelude {
    pub use super::BatchNorm;
}
