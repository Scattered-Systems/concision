/*
    Appellation: sigmoid <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::activate::{ReLU, Sigmoid, Softmax, Tanh};

use ndarray::{Array, ArrayBase, DataMut, DataOwned, Dimension, NdFloat};

impl<A, S, D> ReLU for ArrayBase<S, D>
where
    A: Copy + core::cmp::PartialOrd + num::Zero,
    S: DataMut<Elem = A> + DataOwned,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn relu(&self) -> Self::Output {
        self.map(|&i| if i > A::zero() { i } else { A::zero() })
    }
}

impl<A, S, D> Sigmoid for ArrayBase<S, D>
where
    A: NdFloat,
    S: DataMut<Elem = A> + DataOwned,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn sigmoid(&self) -> Self::Output {
        let dim = self.dim();
        let ones = Array::<A, D>::ones(dim);

        (ones + self.map(|&i| i.neg().exp())).recip()
    }
}

impl<A, S, D> Softmax for ArrayBase<S, D>
where
    A: NdFloat,
    S: DataMut<Elem = A> + DataOwned,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn softmax(&self) -> Self::Output {
        let e = self.exp();
        &e / e.sum()
    }
}

impl<A, S, D> Tanh for ArrayBase<S, D>
where
    A: NdFloat,
    S: DataMut<Elem = A> + DataOwned,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn tanh(&self) -> Self::Output {
        self.map(|i| i.tanh())
    }
}
