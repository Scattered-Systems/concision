/*
    Appellation: perceptron <module>
    Contrib: @FL03
*/
use crate::Predict;
use ndarray::{Array0, Array1, NdFloat, linalg::Dot};
use num::Float;

#[derive(Clone, Debug, PartialEq)]
pub struct Perceptron<T = f64> {
    pub features: usize,
    pub bias: Array0<T>,
    pub weights: Array1<T>,
}

impl<T> Perceptron<T> {
    pub fn ones(features: usize) -> Self
    where
        T: Clone + num::One,
    {
        Self {
            features,
            bias: Array0::ones(()),
            weights: Array1::ones(features),
        }
    }

    pub fn zeros(features: usize) -> Self
    where
        T: Clone + num::Zero,
    {
        Self {
            features,
            bias: Array0::zeros(()),
            weights: Array1::zeros(features),
        }
    }

    #[cfg(feature = "rand")]
    pub fn lecun_normal(features: usize) -> Self
    where
        T: Float,
        rand_distr::StandardNormal: rand_distr::Distribution<T>,
    {
        use crate::init::InitializeExt;

        Self {
            features,
            bias: Array0::lecun_normal((), 1),
            weights: Array1::lecun_normal(features, features),
        }
    }

    pub fn compute_gradient(
        &self,
        input: &Array1<T>,
        target: &Array0<T>,
        lr: T,
    ) -> (Array1<T>, Array0<T>)
    where
        T: Copy + NdFloat,
    {
        let output = self.predict(input).expect("Failed to predict");
        let error = target - output;
        let grad_weights = &error * input * lr;
        let grad_bias = error * lr;
        (grad_weights, grad_bias)
    }

    pub fn apply_gradient(&mut self, grad_weights: &Array1<T>, grad_bias: &Array0<T>)
    where
        T: NdFloat,
    {
        self.weights += grad_weights;
        self.bias += grad_bias;
    }
}

impl<A, B, C, T> Predict<A> for Perceptron<T>
where
    T: NdFloat,
    Array1<T>: Dot<A, Output = B>,
    B: core::ops::Add<T, Output = C>,
{
    type Output = C;

    fn predict(&self, input: &A) -> crate::Result<Self::Output> {
        let res = self.weights.dot(input) + self.bias[()];
        Ok(res)
    }
}
