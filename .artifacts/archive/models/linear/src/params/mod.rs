/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[doc(inline)]
pub use self::item::{Param, Parameter};
pub use self::mode::*;
pub use self::store::*;

mod store;

pub mod item;
pub mod mode;

#[doc(inline)]
pub use crate::primitives::params::*;

pub(crate) mod prelude {
    pub use super::mode::*;
}
