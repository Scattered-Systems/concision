/*
    Appellation: concision-math <library>
    Contrib: @FL03
*/
//! A collection of mathematical functions and utilities for signal processing, statistics, and more.
#![cfg_attr(not(feature = "std"), no_std)]

pub use self::{traits::prelude::*, utils::prelude::*};

#[macro_use]
pub(crate) mod macros;

pub mod signal;
pub mod stats;
pub mod traits;
pub mod utils;

#[allow(unused_imports)]
pub mod prelude {
    pub use crate::signal::prelude::*;
    pub use crate::stats::prelude::*;
    pub use crate::traits::prelude::*;
    pub use crate::utils::prelude::*;
}
