/*
    Appellation: means <module>
    Contrib: @FL03
*/
use ndarray::{ArrayBase, Data, Dimension, NdFloat, ScalarOperand};
use num::traits::{FromPrimitive, Num, Signed};

/// A trait for computing the mean absolute error between two entities
pub trait MeanAbsoluteError<Rhs = Self> {
    type Output;

    fn mae(&self, target: &Rhs) -> Self::Output;
}

/// Compute the mean squared error between two entities
pub trait MeanSquaredError<Rhs = Self> {
    type Output;

    fn mse(&self, target: &Rhs) -> Self::Output;
}

/*
 ************* Implementations *************
*/
impl<A, S, D> MeanAbsoluteError<ArrayBase<S, D>> for ArrayBase<S, D>
where
    A: Copy + FromPrimitive + Num + ScalarOperand + Signed,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Option<A>;

    fn mae(&self, target: &ArrayBase<S, D>) -> Self::Output {
        (target - self).mean().map(|x| x.abs())
    }
}

impl<A, S, D> MeanSquaredError<ArrayBase<S, D>> for ArrayBase<S, D>
where
    A: FromPrimitive + NdFloat,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Option<A>;

    fn mse(&self, target: &ArrayBase<S, D>) -> Self::Output {
        (target - self).pow2().mean()
    }
}
