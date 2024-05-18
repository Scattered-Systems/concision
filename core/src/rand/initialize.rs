/*
    Appellation: initialize <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use core::ops::Neg;
use nd::{ArrayBase, DataOwned, Dimension, RawData, ShapeBuilder};
use ndrand::RandomExt;
use num::traits::Float;
use rand::{rngs, Rng, SeedableRng};
use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Bernoulli, BernoulliError, Distribution, StandardNormal};

/// This trait provides the base methods required for initializing an [ndarray](ndarray::ArrayBase) with random values.
/// [Initialize] is similar to [RandomExt](ndarray_rand::RandomExt), however, it focuses on flexibility while implementing additional
/// features geared towards machine-learning models; such as lecun_normal initialization.
pub trait Initialize<S, D>
where
    D: Dimension,
    S: RawData,
{
    /// Generate a random array using the given distribution
    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> ArrayBase<S, D>
    where
        S: DataOwned,
        Ds: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>;
    /// Generate a random array using the given distribution and random number generator
    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> ArrayBase<S, D>
    where
        R: Rng + ?Sized,
        S: DataOwned,
        Ds: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>;
    /// Initialize an array with random values using the given distribution and current shape
    fn init_rand<Ds>(self, distr: Ds) -> ArrayBase<S, D>
    where
        S: DataOwned,
        Ds: Distribution<S::Elem>,
        Self: Sized;
    /// Initialize an array with random values from the current shape using the given distribution and random number generator
    fn init_rand_with<Ds, R>(self, distr: Ds, rng: &mut R) -> ArrayBase<S, D>
    where
        R: Rng + ?Sized,
        S: DataOwned,
        Ds: Distribution<S::Elem>;
}

/// This trait extends the [Initialize] trait with methods for generating random arrays from various distributions.
pub trait InitializeExt<A, S, D>: Initialize<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn bernoulli<Sh>(shape: Sh, p: Option<f64>) -> Result<ArrayBase<S, D>, BernoulliError>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        Bernoulli: Distribution<A>,
    {
        let dist = Bernoulli::new(p.unwrap_or(0.5))?;
        Ok(Self::rand(shape, dist))
    }
    /// Generate a random array using the [StandardNormal](rand_distr::StandardNormal) distribution
    fn stdnorm<Sh>(shape: Sh) -> ArrayBase<S, D>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        StandardNormal: Distribution<A>,
    {
        Self::rand(shape, StandardNormal)
    }
    /// Generate a random array using the [StandardNormal](rand_distr::StandardNormal) distribution with a given seed
    fn stdnorm_from_seed<Sh>(shape: Sh, seed: u64) -> ArrayBase<S, D>
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        StandardNormal: Distribution<A>,
    {
        Self::rand_with(
            shape,
            StandardNormal,
            &mut rngs::StdRng::seed_from_u64(seed),
        )
    }
    /// A [uniform](rand_distr::uniform::Uniform) generator with values between u(-dk, dk)
    fn uniform<Sh>(shape: Sh, dk: A) -> ArrayBase<S, D>
    where
        A: Clone + Neg<Output = A> + SampleUniform,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::rand(shape, Uniform::new(dk.clone().neg(), dk))
    }
    /// Generate a random array with values between u(-a, a) where a is the reciprocal of the value at the given axis
    fn uniform_along<Sh>(shape: Sh, axis: usize) -> ArrayBase<S, D>
    where
        A: Copy + Float + SampleUniform,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        let dk = A::from(dim[axis]).unwrap().recip();
        Self::uniform(dim, dk)
    }
    /// A [uniform](rand_distr::uniform::Uniform) generator with values between u(-dk, dk)
    fn uniform_between<Sh>(shape: Sh, a: A, b: A) -> ArrayBase<S, D>
    where
        A: SampleUniform,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::rand(shape, Uniform::new(a, b))
    }
}
/*
 ************ Implementations ************
*/
impl<A, S, D> Initialize<S, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
    ArrayBase<S, D>: RandomExt<S, A, D>,
{
    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> ArrayBase<S, D>
    where
        S: DataOwned,
        Ds: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::random(shape, distr)
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> ArrayBase<S, D>
    where
        R: Rng + ?Sized,
        S: DataOwned,
        Ds: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::random_using(shape, distr, rng)
    }

    fn init_rand<Ds>(self, distr: Ds) -> ArrayBase<S, D>
    where
        S: DataOwned,
        Ds: Distribution<S::Elem>,
    {
        Self::rand(self.dim(), distr)
    }

    fn init_rand_with<Ds, R>(self, distr: Ds, rng: &mut R) -> ArrayBase<S, D>
    where
        R: Rng + ?Sized,
        S: DataOwned,
        Ds: Distribution<S::Elem>,
    {
        Self::rand_with(self.dim(), distr, rng)
    }
}

impl<U, A, S, D> InitializeExt<A, S, D> for U
where
    D: Dimension,
    S: RawData<Elem = A>,
    U: Initialize<S, D>,
{
}
