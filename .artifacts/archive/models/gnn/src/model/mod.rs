/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::base::*;

pub(crate) mod base;

pub(crate) mod prelude {
    pub use super::base::GNN;
}
