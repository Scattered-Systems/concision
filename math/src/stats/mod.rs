/*
    Appellation: stats <module>
    Contrib: @FL03
*/
//! Statistical functions and utilities for calculating summary statistics.
//!
pub use self::summary::SummaryStatistics;

pub mod summary;

pub(crate) mod prelude {
    pub use super::summary::*;
}
