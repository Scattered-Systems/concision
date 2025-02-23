/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::ParamsBase;
use concision::prelude::{ModelError, Parameters, Predict};
use core::ops::Add;
use nd::linalg::Dot;
use nd::*;
use num::complex::ComplexFloat;

impl<A, K, S, D> ParamsBase<S, D, K>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    pub fn activate<F, X, Y>(&mut self, args: &X, f: F) -> Y
    where
        F: for<'a> Fn(&'a Y) -> Y,
        S: Data<Elem = A>,
        Self: Predict<X, Output = Y>,
    {
        f(&self.predict(args).unwrap())
    }
}

impl<A, S, D, K> Parameters for ParamsBase<S, D, K>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    type Elem = A;
}

impl<A, S, D> Clone for ParamsBase<S, D>
where
    A: Clone,
    D: RemoveAxis,
    S: RawDataClone<Elem = A>,
{
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            _mode: self._mode,
        }
    }
}

impl<A, S, D> Copy for ParamsBase<S, D>
where
    A: Copy,
    D: Copy + RemoveAxis,
    S: Copy + RawDataClone<Elem = A>,
    <D as Dimension>::Smaller: Copy,
{
}

impl<A, S, D> PartialEq for ParamsBase<S, D>
where
    A: PartialEq,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    fn eq(&self, other: &Self) -> bool {
        self.weights() == other.weight && self.bias == other.bias
    }
}

impl<A, S, D, K> PartialEq<(ArrayBase<S, D>, Option<ArrayBase<S, D::Smaller>>)>
    for ParamsBase<S, D, K>
where
    A: PartialEq,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    fn eq(&self, (weights, bias): &(ArrayBase<S, D>, Option<ArrayBase<S, D::Smaller>>)) -> bool {
        self.weights() == weights && self.bias.as_ref() == bias.as_ref()
    }
}

impl<A, S, D, K> PartialEq<(ArrayBase<S, D>, ArrayBase<S, D::Smaller>)> for ParamsBase<S, D, K>
where
    A: PartialEq,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    fn eq(&self, (weights, bias): &(ArrayBase<S, D>, ArrayBase<S, D::Smaller>)) -> bool {
        self.weights() == weights && self.bias.as_ref() == Some(bias)
    }
}

impl<A, B, T, S, D, K> Predict<A> for ParamsBase<S, D, K>
where
    A: Dot<Array<T, D>, Output = B>,
    B: for<'a> Add<&'a ArrayBase<S, D::Smaller>, Output = B>,
    D: RemoveAxis,
    S: Data<Elem = T>,
    T: ComplexFloat,
{
    type Output = B;

    fn predict(&self, input: &A) -> Result<Self::Output, ModelError> {
        let wt = self.weights().t().to_owned();
        let mut res = input.dot(&wt);
        if let Some(bias) = self.bias.as_ref() {
            res = res + bias;
        }
        Ok(res)
    }
}

impl<'a, A, B, T, S, D, K> Predict<A> for &'a ParamsBase<S, D, K>
where
    A: Dot<Array<T, D>, Output = B>,
    B: Add<&'a ArrayBase<S, D::Smaller>, Output = B>,
    D: RemoveAxis,
    S: Data<Elem = T>,
    T: ComplexFloat,
{
    type Output = B;

    fn predict(&self, input: &A) -> Result<Self::Output, ModelError> {
        let wt = self.weights().t().to_owned();
        let mut res = input.dot(&wt);
        if let Some(bias) = self.bias.as_ref() {
            res = res + bias;
        }
        Ok(res)
    }
}
