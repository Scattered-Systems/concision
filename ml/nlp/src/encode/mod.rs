/*
   Appellation: encode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::utils::*;

pub mod positional;

use ndarray::Dimension;
use ndarray::prelude::{Array, Array2};

pub trait Encode<T> {
    type Output;

    fn encode(&self, data: &T) -> Self::Output;
}

pub trait EncodeArr<T> {
    type Dim: Dimension;

    fn encode(&self, data: &Array<T, Self::Dim>) -> Array2<T>;
}
pub(crate) mod utils {}
