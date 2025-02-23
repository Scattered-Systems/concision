/*
    Appellation: init <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! This module works to provide the crate with various initialization methods suitable for
//! machine-learning models.
//!
//!

pub use self::distr::prelude::*;
pub use self::initialize::*;
pub use self::utils::*;

pub(crate) mod initialize;
pub(crate) mod utils;

pub mod distr {
    pub use self::prelude::*;

    pub mod lecun;
    pub mod trunc;
    pub mod xavier;

    pub(crate) mod prelude {
        pub use super::lecun::*;
        pub use super::trunc::*;
        pub use super::xavier::*;
    }
}

type UniformResult<T = ()> = Result<T, rand_distr::uniform::Error>;

#[doc(hidden)]
#[doc(no_inline)]
pub use rand;
#[doc(no_inline)]
pub use rand_distr;

pub(crate) mod prelude {
    pub use super::distr::prelude::*;
    pub use super::initialize::{Initialize, InitializeExt};
    pub use super::utils::*;
}
