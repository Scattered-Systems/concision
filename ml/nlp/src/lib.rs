/*
   Appellation: concision-nlp <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Natural Language Processing
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod embed;
pub mod encode;

#[cfg(test)]
pub use concision_core as core;

pub mod prelude {
    pub use crate::embed::*;
    pub use crate::encode::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
