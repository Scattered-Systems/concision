/*
    Appellation: impl_rand <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::params::{ParamMode, ParamsBase};
use crate::{bias_dim, Linear};
use concision::rand::rand::Rng;
use concision::rand::rand_distr::{uniform::SampleUniform, Distribution, StandardNormal};
use concision::{Initialize, InitializeExt};
use nd::*;
use num::Float;

impl<A, D, K> Linear<A, K, D>
where
    A: Clone + Float,
    D: RemoveAxis,
    K: ParamMode,
    StandardNormal: Distribution<A>,
{
    pub fn uniform(self) -> Self
    where
        A: SampleUniform,
        <A as SampleUniform>::Sampler: Clone,
    {
        Self {
            params: self.params.uniform(),
            ..self
        }
    }
}

impl<A, K, D> crate::LinearParams<A, K, D>
where
    A: Clone + Float + SampleUniform,
    D: RemoveAxis,
    K: ParamMode,
    StandardNormal: Distribution<A>,
    <A as SampleUniform>::Sampler: Clone,
{
    /// Computes the reciprocal of the input features.
    pub(crate) fn dk(&self) -> A {
        A::from(self.in_features()).unwrap().recip()
    }
    /// Computes the square root of the reciprical of the input features.
    pub(crate) fn dk_sqrt(&self) -> A {
        self.dk().sqrt()
    }

    pub fn uniform(self) -> Self {
        let dk = self.dk_sqrt();
        self.uniform_between(-dk, dk)
    }

    pub fn uniform_between(self, low: A, high: A) -> Self {
        if self.is_biased() && !self.bias.is_some() {
            let b_dim = bias_dim(self.raw_dim());
            Self {
                bias: Some(Array::uniform_between(b_dim, low, high)),
                weight: Array::uniform_between(self.raw_dim(), low, high),
                _mode: self._mode,
            }
        } else if !self.is_biased() && self.bias.is_some() {
            Self {
                bias: None,
                weight: Array::uniform_between(self.raw_dim(), low, high),
                _mode: self._mode,
            }
        } else {
            Self {
                bias: self
                    .bias
                    .as_ref()
                    .map(|b| Array::uniform_between(b.raw_dim(), low, high)),
                weight: Array::uniform_between(self.raw_dim(), low, high),
                _mode: self._mode,
            }
        }
    }
}

impl<A, K, D> Initialize<A, D> for Linear<A, K, D>
where
    D: RemoveAxis,
    K: ParamMode,
    StandardNormal: Distribution<A>,
{
    type Data = OwnedRepr<A>;
    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        Ds: Clone + Distribution<A>,
    {
        Self::from_params(ParamsBase::rand(shape, distr))
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_params(ParamsBase::rand_with(shape, distr, rng))
    }

    fn init_rand<Ds>(self, distr: Ds) -> Self
    where
        Ds: Clone + Distribution<A>,
        Self: Sized,
    {
        Self::rand(self.dim(), distr)
    }

    fn init_rand_with<Ds, R>(self, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Clone + Distribution<A>,
    {
        Self::rand_with(self.dim(), distr, rng)
    }
}

impl<A, S, D, K> Initialize<A, D> for ParamsBase<S, D, K>
where
    D: RemoveAxis,
    K: ParamMode,
    S: DataOwned<Elem = A>,
    StandardNormal: Distribution<A>,
{
    type Data = S;
    fn rand<Sh, Dstr>(shape: Sh, distr: Dstr) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        Dstr: Clone + Distribution<A>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        let bias = if K::BIASED {
            Some(ArrayBase::rand(bias_dim(dim.clone()), distr.clone()))
        } else {
            None
        };
        Self {
            weight: ArrayBase::rand(dim, distr),
            bias,
            _mode: core::marker::PhantomData::<K>,
        }
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        S: DataOwned,
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        let bias = if K::BIASED {
            Some(ArrayBase::rand_with(
                bias_dim(dim.clone()),
                distr.clone(),
                rng,
            ))
        } else {
            None
        };
        Self {
            weight: ArrayBase::rand_with(dim, distr, rng),
            bias,
            _mode: core::marker::PhantomData::<K>,
        }
    }

    fn init_rand<Ds>(self, distr: Ds) -> Self
    where
        S: DataOwned,
        Ds: Clone + Distribution<A>,
        Self: Sized,
    {
        Self::rand(self.dim(), distr)
    }

    fn init_rand_with<Ds, R>(self, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        S: DataOwned,
        Ds: Clone + Distribution<A>,
    {
        Self::rand_with(self.dim(), distr, rng)
    }
}
