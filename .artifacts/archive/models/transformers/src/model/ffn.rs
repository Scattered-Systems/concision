/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::prelude::{Dropout, ModelError, Predict, ReLU};
use linear::{Biased, Linear, ParamMode};
use nd::prelude::*;
use nd::{RemoveAxis, ScalarOperand};
use num::traits::Num;

// #84: FeedForwardNetwork
/// A piecewise, feed-forward neural network consisting of two [Linear] layers with a ReLU activation function
/// optionally (and conditionally) supporting an [Dropout] layer.
///
/// ### Shape
///
/// - d_model: Embedding size
/// - d_ff: upward projection
///
pub struct FeedForwardNetwork<A = f64, K = Biased, D = Ix2>
where
    D: Dimension,
{
    #[cfg(feature = "rand")]
    pub(crate) dropout: Option<Dropout>,
    pub(crate) input: Linear<A, K, D>,
    pub(crate) output: Linear<A, K, D>,
}
#[cfg(not(feature = "rand"))]
impl<A, K> FeedForwardNetwork<A, K, Ix2>
where
    K: ParamMode,
{
    pub fn std(d_model: usize, features: usize) -> Self
    where
        A: Clone + Default,
    {
        let input = Linear::from_features(d_model, features);
        let output = Linear::from_features(features, d_model);
        Self { input, output }
    }
}

#[cfg(feature = "rand")]
impl<A, K> FeedForwardNetwork<A, K, Ix2>
where
    K: ParamMode,
{
    pub fn std(d_model: usize, features: usize, dropout: Option<f64>) -> Self
    where
        A: Clone + Default,
    {
        let dropout = dropout.map(Dropout::new);
        let input = Linear::from_features(d_model, features);
        let output = Linear::from_features(features, d_model);
        Self {
            dropout,
            input,
            output,
        }
    }
}

impl<A, D, K> FeedForwardNetwork<A, K, D>
where
    D: Dimension,
{
    concision::getters!(input, output => Linear<A, K, D>);
}

#[cfg(feature = "rand")]
impl<A, D, K> FeedForwardNetwork<A, K, D>
where
    D: Dimension,
{
    pub fn dropout(&self) -> Option<&Dropout> {
        self.dropout.as_ref()
    }
}

#[cfg(not(feature = "rand"))]
impl<A, D, K> FeedForwardNetwork<A, K, D>
where
    D: Dimension,
{
    pub fn dropout(&self) -> Option<&Dropout> {
        None
    }
}
#[cfg(not(feature = "rand"))]
impl<A, B, D, E, K> Predict<Array<B, E>> for FeedForwardNetwork<A, K, D>
where
    B: Num + PartialOrd + ScalarOperand,
    D: RemoveAxis,
    E: Dimension,
    Linear<A, K, D>: Predict<Array<B, E>, Output = Array<B, E>>,
{
    type Output = Array<B, E>;

    fn predict(&self, input: &Array<B, E>) -> Result<Self::Output, ModelError> {
        let y = self.input().predict(input)?.relu();
        self.output().predict(&y)
    }
}
#[cfg(feature = "rand")]
impl<A, B, D, E, K> Predict<Array<B, E>> for FeedForwardNetwork<A, K, D>
where
    B: Num + PartialOrd + ScalarOperand,
    D: RemoveAxis,
    E: Dimension,
    Linear<A, K, D>: Predict<Array<B, E>, Output = Array<B, E>>,
{
    type Output = Array<B, E>;

    fn predict(&self, input: &Array<B, E>) -> Result<Self::Output, ModelError> {
        let mut y = self.input().predict(input)?.relu();
        if let Some(dropout) = self.dropout() {
            y = dropout.predict(&y)?;
        }
        self.output().predict(&y)
    }
}
