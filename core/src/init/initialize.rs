/*
    Appellation: initialize <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::init::distr::*;

use core::ops::Neg;
use ndarray::{ArrayBase, DataOwned, Dimension, RawData, ShapeBuilder};
use num::complex::ComplexDistribution;
use num::traits::Float;
use rand::{
    Rng, SeedableRng,
    rngs::{SmallRng, StdRng},
};
use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Bernoulli, BernoulliError, Distribution, Normal, NormalError, StandardNormal};

/// This trait facilitates the initialization of tensors with random values. [Initialize]
///
/// This trait provides the base methods required for initializing an [ndarray](ndarray::ArrayBase) with random values.
/// [Initialize] is similar to [RandomExt](ndarray_rand::RandomExt), however, it focuses on flexibility while implementing additional
/// features geared towards machine-learning models; such as lecun_normal initialization.
pub trait Initialize<A, D>
where
    D: Dimension,
{
    type Data: RawData<Elem = A>;

    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;
}

/// This trait extends the [Initialize] trait with methods for generating random arrays from various distributions.
pub trait InitializeExt<A, D>
where
    D: Dimension,
    Self: Initialize<A, D> + Sized,
    Self::Data: DataOwned<Elem = A>,
{
    fn bernoulli<Sh: ShapeBuilder<Dim = D>>(shape: Sh, p: f64) -> Result<Self, BernoulliError>
    where
        Bernoulli: Distribution<A>,
    {
        let dist = Bernoulli::new(p)?;
        Ok(Self::rand(shape, dist))
    }
    /// Initialize the object according to the Lecun Initialization scheme.
    /// LecunNormal distributions are truncated [Normal](rand_distr::Normal)
    /// distributions centered at 0 with a standard deviation equal to the
    /// square root of the reciprocal of the number of inputs.
    fn lecun_normal<Sh: ShapeBuilder<Dim = D>>(shape: Sh, n: usize) -> Self
    where
        A: Float,
        StandardNormal: Distribution<A>,
        Self::Data: DataOwned<Elem = A>,
    {
        let distr = LecunNormal::new(n);
        Self::rand(shape, distr)
    }
    /// Given a shape, mean, and standard deviation generate a new object using the [Normal](rand_distr::Normal) distribution
    fn normal<Sh: ShapeBuilder<Dim = D>>(shape: Sh, mean: A, std: A) -> Result<Self, NormalError>
    where
        A: Float,
        StandardNormal: Distribution<A>,
        Self::Data: DataOwned<Elem = A>,
    {
        let distr = Normal::new(mean, std)?;
        Ok(Self::rand(shape, distr))
    }

    fn randc<Sh: ShapeBuilder<Dim = D>>(shape: Sh, re: A, im: A) -> Self
    where
        ComplexDistribution<A, A>: Distribution<A>,
        Self::Data: DataOwned<Elem = A>,
    {
        let distr = ComplexDistribution::new(re, im);
        Self::rand(shape, &distr)
    }
    /// Generate a random array using the [StandardNormal](rand_distr::StandardNormal) distribution
    fn stdnorm<Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Self
    where
        StandardNormal: Distribution<A>,
        Self::Data: DataOwned<Elem = A>,
    {
        Self::rand(shape, StandardNormal)
    }
    /// Generate a random array using the [StandardNormal](rand_distr::StandardNormal) distribution with a given seed
    fn stdnorm_from_seed<Sh: ShapeBuilder<Dim = D>>(shape: Sh, seed: u64) -> Self
    where
        StandardNormal: Distribution<A>,
        Self::Data: DataOwned<Elem = A>,
    {
        Self::rand_with(shape, StandardNormal, &mut StdRng::seed_from_u64(seed))
    }
    /// Initialize the object using the [TruncatedNormal](crate::init::distr::TruncatedNormal) distribution
    fn truncnorm<Sh: ShapeBuilder<Dim = D>>(shape: Sh, mean: A, std: A) -> Result<Self, NormalError>
    where
        A: Float,
        StandardNormal: Distribution<A>,
        Self::Data: DataOwned<Elem = A>,
    {
        let distr = TruncatedNormal::new(mean, std)?;
        Ok(Self::rand(shape, distr))
    }
    /// A [uniform](rand_distr::uniform::Uniform) generator with values between u(-dk, dk)
    fn uniform<Sh>(shape: Sh, dk: A) -> super::UniformResult<Self>
    where
        A: Copy + Neg<Output = A> + SampleUniform,
        Sh: ShapeBuilder<Dim = D>,
        <A as SampleUniform>::Sampler: Clone,
        Self::Data: DataOwned<Elem = A>,
    {
        Uniform::new(dk.neg(), dk).map(|distr| Self::rand(shape, distr))
    }

    fn uniform_from_seed<Sh>(shape: Sh, start: A, stop: A, key: u64) -> super::UniformResult<Self>
    where
        A: Clone + SampleUniform,
        Sh: ShapeBuilder<Dim = D>,
        <A as SampleUniform>::Sampler: Clone,
        Self::Data: DataOwned<Elem = A>,
    {
        Uniform::new(start, stop)
            .map(|distr| Self::rand_with(shape, distr, &mut StdRng::seed_from_u64(key)))
    }
    /// Generate a random array with values between u(-a, a) where a is the reciprocal of the value at the given axis
    fn uniform_along<Sh>(shape: Sh, axis: usize) -> super::UniformResult<Self>
    where
        A: Copy + Float + SampleUniform,
        Sh: ShapeBuilder<Dim = D>,
        <A as SampleUniform>::Sampler: Clone,
        Self::Data: DataOwned<Elem = A>,
    {
        let dim = shape.into_shape_with_order().raw_dim().clone();
        let dk = A::from(dim[axis]).unwrap().recip();
        Self::uniform(dim, dk)
    }
    /// A [uniform](rand_distr::uniform::Uniform) generator with values between u(-dk, dk)
    fn uniform_between<Sh>(shape: Sh, a: A, b: A) -> super::UniformResult<Self>
    where
        A: Clone + SampleUniform,
        Sh: ShapeBuilder<Dim = D>,
        <A as SampleUniform>::Sampler: Clone,
        Self::Data: DataOwned<Elem = A>,
    {
        Uniform::new(a, b).map(|distr| Self::rand(shape, distr))
    }
}
/*
 ************ Implementations ************
*/

impl<S, A, D> Initialize<A, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Data = S;

    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        Self::rand_with(shape, distr, &mut SmallRng::from_rng(&mut rand::rng()))
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        Self::from_shape_simple_fn(shape, move || distr.sample(rng))
    }
}

impl<U, A, S, D> InitializeExt<A, D> for U
where
    A: Clone,
    D: Dimension,
    S: DataOwned<Elem = A>,
    U: Initialize<A, D, Data = S> + Sized,
{
}
