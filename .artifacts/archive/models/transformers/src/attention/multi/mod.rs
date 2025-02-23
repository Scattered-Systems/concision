/*
    Appellation: multi <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Multi-Head Attention
//!
//!
pub use self::multi_head::*;

// pub(crate) mod config;
pub(crate) mod multi_head;

pub(crate) mod prelude {
    pub use super::multi_head::MultiHeadAttention;
}
