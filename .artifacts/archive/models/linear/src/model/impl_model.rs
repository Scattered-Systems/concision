/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Layout, Linear, LinearConfig};
use crate::{Biased, LinearParams, ParamMode, ParamsBase, Unbiased};
use concision::prelude::{Predict, Result};
use nd::prelude::*;
use nd::{DataOwned, OwnedRepr, RawData, RemoveAxis};

impl<A, K> Linear<A, K, Ix2, OwnedRepr<A>>
where
    K: ParamMode,
{
    pub fn std(inputs: usize, outputs: usize) -> Self
    where
        A: Default,
    {
        let config = LinearConfig::<K, Ix2>::new().with_shape((inputs, outputs));
        let params = ParamsBase::new(config.features());
        Linear { config, params }
    }
}

impl<A, S, D, K> Linear<A, K, D, S>
where
    D: RemoveAxis,
    K: ParamMode,
    S: RawData<Elem = A>,
{
    mbuilder!(new where A: Default, S: DataOwned);
    mbuilder!(ones where A: Clone + num::One, S: DataOwned);
    mbuilder!(zeros where A: Clone + num::Zero, S: DataOwned);

    pub fn from_config(config: LinearConfig<K, D>) -> Self
    where
        A: Clone + Default,
        K: ParamMode,
        S: DataOwned,
    {
        let params = ParamsBase::new(config.dim());
        Self { config, params }
    }

    pub fn from_layout(layout: Layout<D>) -> Self
    where
        A: Clone + Default,
        K: ParamMode,
        S: DataOwned,
    {
        let config = LinearConfig::<K, D>::new().with_layout(layout);
        let params = ParamsBase::new(config.dim());
        Self { config, params }
    }

    pub fn from_params(params: ParamsBase<S, D, K>) -> Self {
        let config = LinearConfig::<K, D>::new().with_shape(params.raw_dim());
        Self { config, params }
    }

    /// Applies an activcation function onto the prediction of the model.
    pub fn activate<X, Y, F>(&self, args: &X, func: F) -> Result<Y>
    where
        F: Fn(Y) -> Y,
        Self: Predict<X, Output = Y>,
    {
        Ok(func(self.predict(args)?))
    }

    pub const fn config(&self) -> &LinearConfig<K, D> {
        &self.config
    }

    pub fn weights(&self) -> &ArrayBase<S, D> {
        self.params.weights()
    }

    pub fn weights_mut(&mut self) -> &mut ArrayBase<S, D> {
        self.params.weights_mut()
    }

    pub const fn params(&self) -> &ParamsBase<S, D, K> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut ParamsBase<S, D, K> {
        &mut self.params
    }

    pub fn into_biased(self) -> Linear<A, Biased, D, S>
    where
        A: Default,
        K: 'static,
        S: DataOwned,
    {
        Linear {
            config: self.config.into_biased(),
            params: self.params.into_biased(),
        }
    }

    pub fn into_unbiased(self) -> Linear<A, Unbiased, D, S>
    where
        A: Default,
        K: 'static,
        S: DataOwned,
    {
        Linear {
            config: self.config.into_unbiased(),
            params: self.params.into_unbiased(),
        }
    }

    pub fn is_biased(&self) -> bool
    where
        K: 'static,
    {
        self.config().is_biased()
    }

    pub fn with_params<E>(self, params: LinearParams<A, K, E>) -> Linear<A, K, E>
    where
        E: RemoveAxis,
    {
        let config = self.config.into_dimensionality(params.raw_dim()).unwrap();
        Linear { config, params }
    }

    pub fn with_name(self, name: impl ToString) -> Self {
        Self {
            config: self.config.with_name(name),
            ..self
        }
    }

    concision::dimensional!(params());
}

impl<A, S, D> Linear<A, Biased, D, S>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    pub fn biased<Sh>(shape: Sh) -> Self
    where
        A: Default,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let config = LinearConfig::<Biased, D>::new().with_shape(shape);
        let params = ParamsBase::biased(config.dim());
        Linear { config, params }
    }

    pub fn bias(&self) -> &ArrayBase<S, D::Smaller> {
        self.params().bias()
    }

    pub fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller> {
        self.params_mut().bias_mut()
    }
}

impl<A, S, D> Linear<A, Unbiased, D, S>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    pub fn unbiased<Sh>(shape: Sh) -> Self
    where
        A: Default,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let config = LinearConfig::<Unbiased, D>::new().with_shape(shape);
        let params = ParamsBase::unbiased(config.dim());
        Linear { config, params }
    }
}
