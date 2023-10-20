/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, statics::*, types::*};

/// Collection of constants used throughout the system
pub(crate) mod constants {
    pub const DEFAULT_EMBEDDING_SIZE: usize = 512;
}

/// Collection of static references used throughout
pub(crate) mod statics {}

/// Collection of types used throughout the system
pub(crate) mod types {}
