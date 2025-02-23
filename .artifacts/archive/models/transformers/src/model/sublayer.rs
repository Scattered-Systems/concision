/*
    Appellation: sublayer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]
use concision::nn::Dropout;
use concision::{ModelError, Predict};
use linear::{Biased, LayerNorm, ParamMode, Unbiased};
use nd::prelude::*;
use nd::{DataOwned, RemoveAxis, ScalarOperand};
use num::traits::{Float, FromPrimitive};

/// A residual connection followed by a [layer norm](LayerNorm)
/// [Transformer](crate::Transformer)
pub struct TransformerSublayer<A = f64, K = Biased, D = Ix2>
where
    D: Dimension,
{
    pub(crate) dropout: Dropout,
    pub(crate) norm: LayerNorm<A, K, D>,
}

impl<A, K, D> TransformerSublayer<A, K, D>
where
    D: RemoveAxis,
{
    pub fn new<Sh>(shape: Sh, dropout: f64) -> Self
    where
        A: Default,
        K: ParamMode,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            dropout: Dropout::new(dropout),
            norm: LayerNorm::new(shape),
        }
    }

    pub fn dropout(&self) -> &Dropout {
        &self.dropout
    }

    pub fn norm(&self) -> &LayerNorm<A, K, D> {
        &self.norm
    }
}

impl<A, S, D> Predict<ArrayBase<S, D>> for TransformerSublayer<A, Unbiased, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: RemoveAxis,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn predict(&self, input: &ArrayBase<S, D>) -> Result<Self::Output, ModelError> {
        let normal = self.norm().predict(input)?;
        let y = input + self.dropout().predict(&normal)?;
        Ok(y)
    }
}

impl<A, S, D> Predict<ArrayBase<S, D>> for TransformerSublayer<A, Biased, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: RemoveAxis,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn predict(&self, input: &ArrayBase<S, D>) -> Result<Self::Output, ModelError> {
        let normal = self.norm().predict(input)?;
        let y = input + self.dropout().predict(&normal)?;
        Ok(y)
    }
}
