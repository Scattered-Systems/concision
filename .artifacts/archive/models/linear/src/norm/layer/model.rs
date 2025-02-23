/*
    Appellation: layer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Config;
use crate::{Biased, LinearParams, ParamMode, Unbiased};
use concision::{ModelError, Predict};
use nd::prelude::*;
use nd::{Data, RemoveAxis};
use num::traits::{Float, FromPrimitive, One, Zero};

// #62
///
/// Layer Normalization directly estimates the normalization statistics from the summed inputs
/// to the neurons within a _hidden_ layer, eliminating the need to introduce any additional dependencies.
///
/// [LayerNorm] follows the [Layer Normalization](https://arxiv.org/abs/1607.06450) paper.
///
/// ### Resources
pub struct LayerNorm<A = f64, K = crate::Biased, D = Ix2>
where
    D: Dimension,
{
    config: Config<D>,
    params: LinearParams<A, K, D>,
}

macro_rules! impl_norm_builder {
    ($method:ident$(.$call:ident)? where $($rest:tt)*) => {
        impl_norm_builder!(@impl $method$(.$call)? where $($rest)*);
    };
    (@impl $method:ident where $($rest:tt)*) => {
        impl_norm_builder!(@impl $method.$method where $($rest)*);
    };
    (@impl $method:ident.$call:ident where $($rest:tt)*) => {
        pub fn $method<Sh>(shape: Sh) -> Self
        where
            Sh: ShapeBuilder<Dim = D>,
            $($rest)*
        {
            Self::from_params(LinearParams::<A, K, D>::$call(shape))
        }
    };
}

impl<A, K, D> LayerNorm<A, K, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    pub fn from_config(config: Config<D>) -> Self
    where
        A: Default,
    {
        let params = LinearParams::<A, K, D>::new(config.dim());
        Self { config, params }
    }

    pub fn from_elem<Sh>(shape: Sh, elem: A) -> Self
    where
        A: Clone,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape_with_order().raw_dim().clone();
        let config = Config::new().dim(dim.clone()).build();
        let params = LinearParams::<A, K, D>::from_elem(dim, elem);
        Self { config, params }
    }

    pub fn from_params(params: LinearParams<A, K, D>) -> Self {
        let config = Config::new().dim(params.raw_dim()).build();
        Self { config, params }
    }

    impl_norm_builder!(new where A: Default);
    impl_norm_builder!(ones where A: Clone + One);
    impl_norm_builder!(zeros where A: Clone + Zero);

    pub const fn config(&self) -> &Config<D> {
        &self.config
    }

    pub fn is_biased(&self) -> bool {
        self.params().is_biased()
    }
    /// Returns an immutable reference to the layer's parameters.
    pub const fn params(&self) -> &LinearParams<A, K, D> {
        &self.params
    }
    /// Returns a mutable reference to the layer's parameters.
    pub fn params_mut(&mut self) -> &mut LinearParams<A, K, D> {
        &mut self.params
    }
    /// Returns the epsilon value used for numerical stability.
    pub const fn eps(&self) -> f64 {
        self.config().eps()
    }

    concision::dimensional!(config());
}

impl<A, D> Default for LayerNorm<A, Biased, D>
where
    A: Default,
    D: RemoveAxis,
{
    fn default() -> Self {
        Self {
            config: Config::default(),
            params: Default::default(),
        }
    }
}

impl<A, D> Default for LayerNorm<A, Unbiased, D>
where
    A: Default,
    D: RemoveAxis,
{
    fn default() -> Self {
        Self {
            config: Config::default(),
            params: Default::default(),
        }
    }
}

impl<A, S, D> Predict<ArrayBase<S, D>> for LayerNorm<A, Biased, D>
where
    A: Float + FromPrimitive,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn predict(&self, x: &ArrayBase<S, D>) -> Result<Self::Output, ModelError> {
        let norm = if let Some(axis) = self.config().axis() {
            super::layer_norm_axis(x, *axis, self.eps())
        } else {
            super::layer_norm(x, self.eps())
        };
        let y = norm * self.params().weights() + self.params().bias();
        Ok(y)
    }
}

impl<A, S, D> Predict<ArrayBase<S, D>> for LayerNorm<A, Unbiased, D>
where
    A: Float + FromPrimitive,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn predict(&self, x: &ArrayBase<S, D>) -> Result<Self::Output, ModelError> {
        let norm = if let Some(axis) = self.config().axis() {
            super::layer_norm_axis(x, *axis, self.eps())
        } else {
            super::layer_norm(x, self.eps())
        };
        let y = norm * self.params().weights();
        Ok(y)
    }
}
