/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::kan::*;

pub(crate) mod kan;

pub(crate) mod prelude {
    pub use super::kan::KAN;
}
