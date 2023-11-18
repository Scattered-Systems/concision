/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::GenerateRandom;

use crate::prelude::{Biased, Weighted};
use ndarray::prelude::{Array0, Array1, Array2, Ix1, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::{Float, FromPrimitive};

#[derive(Clone, Debug, PartialEq)]
pub struct NeuronParams<T = f64>
where
    T: Float,
{
    bias: Array0<T>,
    pub features: usize,
    weights: Array1<T>,
}

impl<T> NeuronParams<T>
where
    T: Float,
{
    pub fn new(features: usize) -> Self {
        Self {
            bias: Array0::zeros(()),
            features,
            weights: Array1::zeros(features),
        }
    }

    pub fn features(&self) -> usize {
        self.features
    }

    pub fn features_mut(&mut self) -> &mut usize {
        &mut self.features
    }

    pub fn set_features(&mut self, features: usize) {
        self.features = features;
    }

    pub fn with_bias(mut self, bias: Array0<T>) -> Self {
        self.bias = bias.into();
        self
    }

    pub fn with_features(mut self, features: usize) -> Self {
        self.features = features;
        self
    }

    pub fn with_weights(mut self, weights: Array1<T>) -> Self {
        self.weights = weights;
        self
    }
}

impl<T> NeuronParams<T>
where
    T: Float + SampleUniform,
{
    pub fn init(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dk = (T::one() / T::from(self.features).unwrap()).sqrt();
        self.bias = Array0::uniform_between(dk, ());
        self
    }

    pub fn init_weight(mut self) -> Self {
        let features = self.features;
        let dk = (T::one() / T::from(features).unwrap()).sqrt();
        self.weights = Array1::uniform_between(dk, features);
        self
    }
}

impl<T> NeuronParams<T>
where
    T: FromPrimitive + NdFloat,
{
    pub fn apply_gradient<G>(&mut self, gamma: T, gradient: G)
    where
        G: Fn(&Array1<T>) -> Array1<T>,
    {
        let mut grad = gradient(self.weights());
        grad /= grad.mapv(|ws| ws.powi(2)).sum().sqrt();
        self.weights_mut().scaled_add(-gamma, &grad);
        self.weights /= self.weights().mapv(|ws| ws.powi(2)).sum().sqrt();
    }

    pub fn linear(&self, data: &Array2<T>) -> Array1<T> {
        data.dot(&self.weights().t()) + self.bias()
    }
}

// impl<T> Params<T, Ix1> for NeuronParams<T>
// where
//     T: Float,
// {
//     fn bias(&self) -> &Array0<T> {
//         &self.bias
//     }

//     fn bias_mut(&mut self) -> &mut Array0<T> {
//         &mut self.bias
//     }

//     fn weights(&self) -> &Array1<T> {
//         &self.weights
//     }

//     fn weights_mut(&mut self) -> &mut Array1<T> {
//         &mut self.weights
//     }

//     fn set_bias(&mut self, bias: Array0<T>) {
//         self.bias = bias;
//     }

//     fn set_weights(&mut self, weights: Array1<T>) {
//         self.weights = weights;
//     }
// }

impl<T> Biased<T, Ix1> for NeuronParams<T>
where
    T: Float,
{
    fn bias(&self) -> &Array0<T> {
        &self.bias
    }

    fn bias_mut(&mut self) -> &mut Array0<T> {
        &mut self.bias
    }

    fn set_bias(&mut self, bias: Array0<T>) {
        self.bias = bias;
    }
}

impl<T> Weighted<T, Ix1> for NeuronParams<T>
where
    T: Float,
{
    fn set_weights(&mut self, weights: Array1<T>) {
        self.weights = weights;
    }

    fn weights(&self) -> &Array1<T> {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut Array1<T> {
        &mut self.weights
    }
}
