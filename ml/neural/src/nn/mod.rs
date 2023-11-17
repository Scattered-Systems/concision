/*
    Appellation: nn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Neural Network
pub use self::{deep::*, utils::*};

pub(crate) mod deep;

use crate::layers::Layer;
use crate::Trainable;
use num::Float;

pub trait NeuralNet<T: Float = f64>: Trainable<T> {
    fn depth(&self) -> usize {
        self.layers().len()
    }

    fn layers(&self) -> &[Layer<T>];

    fn input_layer(&self) -> &Layer<T> {
        &self.layers()[0]
    }
}

pub(crate) mod utils {}
