/*
    Appellation: actor <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::config::ActorConfig;

pub(crate) mod config;

use concision::prelude::{Eval, ModelError, Predict};
use core::ops::Mul;
use nd::prelude::*;
use splines::Spline;
use splines::interpolate::{Interpolate, Interpolator};

#[doc(hidden)]
pub type NdSpline<T, V> = Spline<Array1<T>, Array1<V>>;

/// An `actor` represents the learned activation functions
/// described within the KAN paper.
///
/// The learned activation functions are univariate and continuous.
///
/// ### Parameters
///
/// - `b(**x**)`: bias function
/// - spline(**x**): spline function; typically employs `Linear` interpolation
/// - **ω**: weight factor
///
/// The learned functions are one-dimensional, continuous
pub struct Actor<T, V, B> {
    pub(crate) bias: B,
    pub(crate) spline: Spline<T, V>,
    pub(crate) weight: Array1<V>,
}

impl<T, V, B> Actor<T, V, B> {
    pub fn new(bias: B, spline: Spline<T, V>, weight: Array1<V>) -> Self {
        Self {
            bias,
            spline,
            weight,
        }
    }

    pub fn bias(&self) -> &B {
        &self.bias
    }

    pub fn sample(&self, x: T) -> Option<V>
    where
        T: Interpolator,
        V: Interpolate<T>,
    {
        self.spline().sample(x)
    }

    pub fn spline(&self) -> &Spline<T, V> {
        &self.spline
    }

    pub fn spline_mut(&mut self) -> &mut Spline<T, V> {
        &mut self.spline
    }

    pub fn weight(&self) -> &Array1<V> {
        &self.weight
    }

    pub fn weight_mut(&mut self) -> &mut Array1<V> {
        &mut self.weight
    }
}

impl<Z, T, V, B> Predict<Array1<T>> for Actor<T, V, B>
where
    B: Eval<Array1<V>>,
    T: Interpolator,
    V: Interpolate<T>,
    for<'a> &'a Array1<V>: Mul<B::Output, Output = Z>,
{
    type Output = Z;

    fn predict(&self, x: &Array1<T>) -> Result<Self::Output, ModelError> {
        let y = x.mapv(|xi| self.spline().sample(xi).unwrap());
        Ok(self.weight() * self.bias().eval(y))
    }
}
