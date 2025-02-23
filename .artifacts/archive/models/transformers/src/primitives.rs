/*
    Appellation: primitives <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::consts::*;

pub mod consts {
    /// The default dimension of the model; i.e. the number of inputs
    pub const D_MODEL: usize = 512;
    /// The default size of the network; i.e. the number of neurons in the network
    pub const D_NETWORK: usize = 2048;
    /// The default dimension of the key and query vectors
    pub const DK: usize = D_MODEL / HEADS;
    /// The default number of attention heads
    pub const HEADS: usize = 8;
    /// The default number of layers used for the encoder / decoder.
    pub const N: usize = 6;
}

pub fn outputs_from_ratio(model: usize, network: usize) -> usize {
    network / model
}
