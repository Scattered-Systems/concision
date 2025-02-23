/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub use self::store::*;

pub(crate) mod store;

pub(crate) mod prelude {
    pub use super::store::GraphStore;
}
