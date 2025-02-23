/*
    Appellation: traits <activate>
    Contrib: @FL03
*/

use super::utils::*;

use ndarray::prelude::*;
use ndarray::{Data, DataMut, RemoveAxis, ScalarOperand};
use num::complex::ComplexFloat;
use num::traits::{One, Zero};

unary! {
    Heavyside::heavyside(self),
    LinearActivation::linear(self),
    Sigmoid::sigmoid(&self),
    Softmax::softmax(&self),
    ReLU::relu(&self),
    Tanh::tanh(&self),
}

pub trait SoftmaxAxis: Softmax {
    fn softmax_axis(self, axis: usize) -> Self::Output;
}

pub trait NdActivate<A, D>
where
    A: ScalarOperand,
    D: Dimension,
{
    type Data: Data<Elem = A>;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B;

    fn activate_inplace<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        F: FnMut(A) -> A,
        Self::Data: DataMut<Elem = A>;

    fn linear(&self) -> Array<A, D>
    where
        A: Clone,
    {
        self.activate(|x| x.clone())
    }

    fn heavyside(&self) -> Array<A, D>
    where
        A: One + PartialOrd + Zero,
    {
        self.activate(heavyside)
    }

    fn relu(&self) -> Array<A, D>
    where
        A: PartialOrd + Zero,
    {
        self.activate(relu)
    }

    fn sigmoid(&self) -> Array<A, D>
    where
        A: NdFloat,
    {
        self.activate(sigmoid)
    }

    fn softmax(&self) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
    {
        let exp = self.activate(A::exp);
        &exp / exp.sum()
    }

    fn softmax_axis(&self, axis: usize) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        D: RemoveAxis,
    {
        let exp = self.activate(A::exp);
        let axis = Axis(axis);
        &exp / &exp.sum_axis(axis)
    }

    fn tanh(&self) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
    {
        self.activate(A::tanh)
    }
}
/*
 ************* Implementations *************
*/

impl<A, S, D> NdActivate<A, D> for ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Data = S;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B,
    {
        self.mapv(f)
    }

    fn activate_inplace<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        S: DataMut,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}
