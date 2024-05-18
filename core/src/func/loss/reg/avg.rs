/*
    Appellation: avg <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::math::{Abs, Squared};
use nd::prelude::*;
use nd::{Data, ScalarOperand};
use num::traits::{FromPrimitive, Num, Pow, Signed};

pub fn mae<A, S, D>(pred: &ArrayBase<S, D>, target: &ArrayBase<S, D>) -> Option<A>
where
    A: FromPrimitive + Num + ScalarOperand + Signed,
    D: Dimension,
    S: Data<Elem = A>,
{
    (pred - target).abs().mean()
}

pub fn mse<A, S, D>(pred: &ArrayBase<S, D>, target: &ArrayBase<S, D>) -> Option<A>
where
    A: FromPrimitive + Num + Pow<i32, Output = A> + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    (pred - target).sqrd().mean()
}

pub trait MeanAbsoluteError<Rhs = Self> {
    type Output;

    fn mae(&self, target: &Rhs) -> Self::Output;
}

pub trait MeanSquaredError<Rhs = Self> {
    type Output;

    fn mse(&self, target: &Rhs) -> Self::Output;
}

impl<A, S, D> MeanAbsoluteError<ArrayBase<S, D>> for ArrayBase<S, D>
where
    A: FromPrimitive + Num + ScalarOperand + Signed,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Option<A>;

    fn mae(&self, target: &ArrayBase<S, D>) -> Self::Output {
        (self - target).abs().mean()
    }
}

impl<A, S, D> MeanSquaredError<ArrayBase<S, D>> for ArrayBase<S, D>
where
    A: FromPrimitive + Num + Pow<i32, Output = A> + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Option<A>;

    fn mse(&self, target: &ArrayBase<S, D>) -> Self::Output {
        (self - target).sqrd().mean()
    }
}
