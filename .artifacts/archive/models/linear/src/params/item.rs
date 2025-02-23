/*
    Appellation: entry <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::ParamsBase;
use core::marker::PhantomData;
use ndarray::*;
use strum::{AsRefStr, EnumDiscriminants, EnumIs, VariantNames};

#[derive(AsRefStr, EnumDiscriminants, EnumIs, VariantNames)]
#[cfg_attr(
    feature = "serde",
    strum_discriminants(
        derive(serde::Deserialize, serde::Serialize),
        serde(rename_all = "lowercase", untagged),
    )
)]
#[non_exhaustive]
#[strum(serialize_all = "lowercase")]
#[strum_discriminants(
    name(Param),
    derive(
        AsRefStr,
        Hash,
        Ord,
        PartialOrd,
        strum::Display,
        strum::EnumCount,
        strum::EnumIs,
        strum::EnumIter,
        strum::EnumString,
        strum::VariantArray,
        strum::VariantNames,
    ),
    strum(serialize_all = "lowercase")
)]
pub enum Parameter<S, D>
where
    S: RawData,
    D: RemoveAxis,
{
    Bias(ArrayBase<S, D::Smaller>),
    Weight(ArrayBase<S, D>),
}

impl<A, S, D> Parameter<S, D>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    pub fn from_bias(data: ArrayBase<S, D::Smaller>) -> Self {
        Self::Bias(data)
    }

    pub fn from_weight(data: ArrayBase<S, D>) -> Self {
        Self::Weight(data)
    }
}

impl Param {
    pub fn bias() -> Self {
        Self::Bias
    }

    pub fn weight() -> Self {
        Self::Weight
    }
}

pub struct Item<S, D, E>
where
    D: Dimension<Smaller = E>,
    E: RemoveAxis,
    S: RawData,
{
    pub data: ParamsBase<S, E>,
    _parent: PhantomData<D>,
}
