/*
   Appellation: base <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Axis, Dimension, Ix2};
use ndarray::{IntoDimension, ScalarOperand, ShapeError};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::uniform::{SampleUniform, Uniform};
use ndarray_rand::rand_distr::{Bernoulli, BernoulliError, Distribution, StandardNormal};
use ndarray_rand::RandomExt;
use num::traits::real::Real;
use num::traits::{Float, Num, NumAssignOps, Signed};
use std::ops;

pub trait Affine<T = f64>: Sized {
    type Error;

    fn affine(&self, mul: T, add: T) -> Result<Self, Self::Error>;
}

impl<S, T, D> Affine<S> for Array<T, D>
where
    T: Num + ScalarOperand,
    D: Dimension,
    Array<T, D>: ops::Mul<S, Output = Array<T, D>> + ops::Add<S, Output = Array<T, D>>,
{
    type Error = ShapeError;

    fn affine(&self, mul: S, add: S) -> Result<Self, Self::Error> {
        Ok(self.clone() * mul + add)
    }
}

pub trait GenerateRandom<T = f64, D = Ix2>: Sized
where
    D: Dimension,
{
    fn rand<IdS>(dim: impl IntoDimension<Dim = D>, distr: IdS) -> Self
    where
        IdS: Distribution<T>;

    fn rand_using<IdS, R: ?Sized>(
        dim: impl IntoDimension<Dim = D>,
        distr: IdS,
        rng: &mut R,
    ) -> Self
    where
        IdS: Distribution<T>,
        R: Rng;

    fn bernoulli(dim: impl IntoDimension<Dim = D>, p: Option<f64>) -> Result<Self, BernoulliError>
    where
        Bernoulli: Distribution<T>,
    {
        let dist = Bernoulli::new(p.unwrap_or(0.5))?;
        Ok(Self::rand(dim.into_dimension(), dist))
    }

    fn stdnorm(dim: impl IntoDimension<Dim = D>) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        Self::rand(dim, StandardNormal)
    }

    fn normal_from_key<R: ?Sized>(key: u64, dim: impl IntoDimension<Dim = D>) -> Self
    where
        StandardNormal: Distribution<T>,
        R: Rng,
    {
        Self::rand_using(
            dim.into_dimension(),
            StandardNormal,
            &mut StdRng::seed_from_u64(key),
        )
    }

    fn uniform(axis: usize, dim: impl IntoDimension<Dim = D>) -> Self
    where
        T: Real + SampleUniform,
    {
        let dim = dim.into_dimension();
        let dk = T::from(dim[axis]).unwrap().recip().sqrt();
        Self::uniform_between(dk, dim)
    }

    fn uniform_between(dk: T, dim: impl IntoDimension<Dim = D>) -> Self
    where
        T: SampleUniform + Signed,
    {
        Self::rand(dim, Uniform::new(-dk, dk))
    }
}

impl<T, D> GenerateRandom<T, D> for Array<T, D>
where
    T: Float + SampleUniform,
    D: Dimension,
    StandardNormal: Distribution<T>,
{
    fn rand<IdS>(dim: impl IntoDimension<Dim = D>, distr: IdS) -> Self
    where
        IdS: Distribution<T>,
    {
        Self::random(dim.into_dimension(), distr)
    }

    fn rand_using<IdS, R: ?Sized>(dim: impl IntoDimension<Dim = D>, distr: IdS, rng: &mut R) -> Self
    where
        IdS: Distribution<T>,
        R: Rng,
    {
        Self::random_using(dim.into_dimension(), distr, rng)
    }
}

pub trait IntoAxis {
    fn into_axis(self) -> Axis;
}

impl<S> IntoAxis for S
where
    S: AsRef<usize>,
{
    fn into_axis(self) -> Axis {
        Axis(*self.as_ref())
    }
}

pub trait Inverse<T = f64>: Sized {
    fn inverse(&self) -> Option<Self>;
}

impl<T> Inverse<T> for Array<T, Ix2>
where
    T: Copy + Num + NumAssignOps + ScalarOperand,
{
    fn inverse(&self) -> Option<Self> {
        super::utils::inverse(self)
    }
}

// pub trait Stack

pub trait ArrayLike {
    fn ones_like(&self) -> Self;

    fn zeros_like(&self) -> Self;
}

impl<T, D> ArrayLike for Array<T, D>
where
    T: Clone + Num,
    D: Dimension,
{
    fn ones_like(&self) -> Self {
        Array::ones(self.dim())
    }

    fn zeros_like(&self) -> Self {
        Array::zeros(self.dim())
    }
}
