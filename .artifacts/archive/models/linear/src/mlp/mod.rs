/*
    Appellation: mlp <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Multi-Layer Perceptron (MLP)
//!
//! A multi-layer perceptron (MLP) is a class of feed-forward artificial neural networks (FFN).
//!
//!
#[doc(inline)]
pub use self::{model::*, perceptron::*};

pub(crate) mod model;
pub(crate) mod perceptron;

pub(crate) mod prelude {
    pub use super::perceptron::Perceptron;
}

pub trait DeepNeuralNetwork<T> {
    type Input;
    type Output;
}
