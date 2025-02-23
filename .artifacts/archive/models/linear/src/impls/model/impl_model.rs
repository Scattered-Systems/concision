/*
    Appellation: impl_model <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Linear, LinearConfig, ParamsBase};
use concision::prelude::{ModelError, Module, Predict};
use nd::{RawData, RemoveAxis};

impl<A, D, S, K> Module for Linear<A, K, D, S>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    type CSpace = LinearConfig<K, D>;
    type Elem = A;
    type Params = ParamsBase<S, D, K>;

    fn config(&self) -> &Self::CSpace {
        &self.config
    }

    fn params(&self) -> &Self::Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Self::Params {
        &mut self.params
    }
}

impl<U, V, A, S, D, K> Predict<U> for Linear<A, K, D, S>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
    ParamsBase<S, D, K>: Predict<U, Output = V>,
{
    type Output = V;

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, level = "debug", name = "predict", target = "linear")
    )]
    fn predict(&self, input: &U) -> Result<Self::Output, ModelError> {
        #[cfg(feature = "tracing")]
        tracing::debug!("Predicting with linear model");
        self.params().predict(input)
    }
}
