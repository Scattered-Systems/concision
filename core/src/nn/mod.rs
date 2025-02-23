/*
   Appellation: nn <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//!
pub use self::{dropout::Dropout, model::prelude::*};

pub mod dropout;
pub mod mask;
pub mod model;

#[allow(unused_imports)]
pub(crate) mod prelude {
    pub use super::dropout::Dropout;
    pub use super::mask::Mask;
    pub use super::model::prelude::*;
}
