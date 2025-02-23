/*
    Appellation: params <module>
    Contrib: @FL03
*/
use ndarray::{ArrayBase, Axis, DataOwned, Dimension, RawData, RemoveAxis, linalg::Dot};

use crate::Predict;

pub struct ModelParams<S, D = ndarray::Ix1>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) bias: ArrayBase<S, D::Smaller>,
    pub(crate) weights: ArrayBase<S, D>,
}

impl<A, S, D> ModelParams<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + num::One,
        D: RemoveAxis,
        S: DataOwned,
        Sh: ndarray::ShapeBuilder<Dim = D>,
    {
        let weights = ArrayBase::ones(shape);
        let dim = weights.raw_dim();
        let bias = ArrayBase::from_elem(dim.remove_axis(Axis(0)), A::one());
        Self { bias, weights }
    }
    pub fn bias(&self) -> &ArrayBase<S, D::Smaller> {
        &self.bias
    }

    pub fn weights(&self) -> &ArrayBase<S, D> {
        &self.weights
    }
}

impl<A, S> ModelParams<S>
where
    S: RawData<Elem = A>,
{
    pub fn new(bias: A, weights: ArrayBase<S, ndarray::Ix1>) -> Self
    where
        A: Clone,
        S: DataOwned,
    {
        Self {
            bias: ArrayBase::from_elem((), bias),
            weights,
        }
    }
}

impl<A, B, C, S, D> Predict<B> for ModelParams<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
    ArrayBase<S, D>: Dot<B, Output = C>,
    C: for<'a> core::ops::Add<&'a ArrayBase<S, D::Smaller>, Output = C>,
{
    type Output = C;

    fn predict(&self, input: &B) -> crate::Result<Self::Output> {
        let res = self.weights.dot(input) + &self.bias;
        Ok(res)
    }
}
