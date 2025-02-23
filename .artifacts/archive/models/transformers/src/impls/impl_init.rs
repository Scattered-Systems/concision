/*
    Appellation: init <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]
use crate::QkvBase;
use concision::Initialize;
use concision::init::rand::Rng;
use concision::init::rand_distr::{Distribution, StandardNormal};
use concision::init::rand_distr::uniform::SampleUniform;
use nd::{ArrayBase, DataOwned, Dimension, ShapeBuilder};

impl<A, S, D> Initialize for QkvBase<S, D> where
    D: RemoveAxis,
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
        Self {
            q: ArrayBase::rand(dim.clone(), distr.clone()),
            k: ArrayBase::rand(dim.clone(), distr.clone()),
            v: ArrayBase::rand(dim, distr)
        }
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
        Ds: Clone + Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        Self {
            q: ArrayBase::rand_with(dim.clone(), distr.clone(), &mut rng),
            k: ArrayBase::rand_with(dim.clone(), distr.clone(), &mut rng),
            v: ArrayBase::rand_with(dim, distr, &mut rng)
        }
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


