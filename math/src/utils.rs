/*
    Appellation: utils <module>
    Contrib: @FL03
*/
//! # Math Utilities
#[doc(inline)]
pub use self::prelude::*;

pub mod arith;

pub(crate) mod prelude {
    pub use super::arith::*;
}
