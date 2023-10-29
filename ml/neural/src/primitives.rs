/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, statics::*, types::*};

pub(crate) mod constants {
    pub const DEFAULT_BUFFER: usize = 1024;
}

pub(crate) mod statics {}

pub(crate) mod types {

    pub type BoxedFunction<T> = Box<dyn Fn(T) -> T>;

    pub type LayerBias<T = f64> = ndarray::Array1<T>;

    pub type WeightTensor<T = f64> = ndarray::Array2<T>;
}
