/*
    Appellation: layer <module>
    Contrib: @FL03
*/
use crate::{Features, Perceptron};

pub struct Layer<T> {
    pub index: usize,
    pub features: Features,
    pub nodes: Vec<Perceptron<T>>,
}

impl<T> Layer<T> {
    pub fn zeros(index: usize, features: Features) -> Self
    where
        T: Clone + num::Zero,
    {
        let nodes = (0..features.inputs())
            .map(|_| Perceptron::zeros(features.outputs()))
            .collect::<Vec<_>>();
        Self {
            index,
            features,
            nodes,
        }
    }

    pub fn features(&self) -> &Features {
        &self.features
    }

    pub fn nodes(&self) -> &[Perceptron<T>] {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut [Perceptron<T>] {
        &mut self.nodes
    }
}
