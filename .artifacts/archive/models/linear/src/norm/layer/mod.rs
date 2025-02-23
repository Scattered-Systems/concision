/*
    Appellation: layer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Layer Normalization
//!
//! This module provides the necessary tools for creating and training layer normalization layers.
pub(crate) use self::utils::*;
pub use self::{config::*, model::*};

pub(crate) mod config;
pub(crate) mod model;

pub const EPSILON: f64 = 1e-5;

pub(crate) mod prelude {
    pub use super::config::Config as LayerNormConfig;
    pub use super::model::LayerNorm;
}

pub(crate) mod utils {
    use nd::prelude::*;
    use nd::{Data, RemoveAxis};
    use num::traits::{Float, FromPrimitive};

    pub(crate) fn layer_norm<A, S, D>(x: &ArrayBase<S, D>, eps: f64) -> Array<A, D>
    where
        A: Float + FromPrimitive,
        D: Dimension,
        S: Data<Elem = A>,
    {
        let mean = x.mean().unwrap();
        let denom = {
            let eps = A::from(eps).unwrap();
            let var = x.var(A::zero());
            (var + eps).sqrt()
        };
        x.mapv(|xi| (xi - mean) / denom)
    }

    pub(crate) fn layer_norm_axis<A, S, D>(x: &ArrayBase<S, D>, axis: Axis, eps: f64) -> Array<A, D>
    where
        A: Float + FromPrimitive,
        D: RemoveAxis,
        S: Data<Elem = A>,
    {
        let eps = A::from(eps).unwrap();
        let mean = x.mean_axis(axis).unwrap();
        let var = x.var_axis(axis, A::zero());
        let inv_std = var.mapv(|v| (v + eps).recip().sqrt());
        let x_norm = (x - &mean) * &inv_std;
        x_norm
    }
}
