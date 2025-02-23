/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::layout::prelude::*;

mod impl_config;
mod impl_model;

pub mod layout {
    pub use self::{features::*, layout::*};

    mod features;
    mod layout;

    pub(crate) mod prelude {
        pub use super::features::Features;
        pub use super::layout::Layout;
    }
}

pub(crate) mod prelude {
    pub use super::Linear;
}

use crate::{Biased, ParamsBase};
use ndarray::{Dimension, Ix2, OwnedRepr, RawData};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct LinearConfig<K = Biased, D = Ix2> {
    pub layout: Layout<D>,
    pub name: String,
    _biased: core::marker::PhantomData<K>,
}

/// An implementation of a linear model.
///
/// In an effort to streamline the api, the [Linear] model relies upon a [ParamMode] type ([Biased] or [Unbiased](crate::params::mode::Unbiased))
/// which enables the model to automatically determine whether or not to include a bias term. Doing so allows the model to inherit several methods
/// familar to the underlying [ndarray](https://docs.rs/ndarray) crate.
pub struct Linear<A = f64, K = Biased, D = Ix2, S = OwnedRepr<A>>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub(crate) config: LinearConfig<K, D>,
    pub(crate) params: ParamsBase<S, D, K>,
}
