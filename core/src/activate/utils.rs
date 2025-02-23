/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis, ScalarOperand};
use num::traits::{Float, One, Zero};

/// Heaviside activation function
pub fn heavyside<T>(x: T) -> T
where
    T: One + PartialOrd + Zero,
{
    if x > T::zero() { T::one() } else { T::zero() }
}
///
pub fn relu<T>(args: T) -> T
where
    T: PartialOrd + Zero,
{
    if args > T::zero() { args } else { T::zero() }
}
///
pub fn sigmoid<T>(args: T) -> T
where
    T: Float,
{
    (T::one() + args.neg().exp()).recip()
}
///
pub fn softmax<A, S, D>(args: &ArrayBase<S, D>) -> Array<A, D>
where
    A: Float + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    let e = args.exp();
    &e / e.sum()
}
///
pub fn softmax_axis<A, S, D>(args: &ArrayBase<S, D>, axis: usize) -> Array<A, D>
where
    A: Float + ScalarOperand,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    let axis = Axis(axis);
    let e = args.exp();
    &e / &e.sum_axis(axis)
}
///
pub fn tanh<T>(args: T) -> T
where
    T: num::traits::Float,
{
    args.tanh()
}
