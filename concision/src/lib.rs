/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision
//!
//! Concision aims to provide a concise and efficient interface for machine learning and data processing.
#![allow(unused_imports)]
#![crate_name = "concision"]

#[doc(inline)]
pub use concision_core::*;
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "macros")]
pub use concision_macros::*;

pub mod prelude {
    pub use concision_core::prelude::*;
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
}
