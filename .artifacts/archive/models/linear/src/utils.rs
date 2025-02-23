/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::Biased;
use nd::{ArrayBase, Axis, Dimension, RawData, RemoveAxis};

/// A utilitarian funciton for building bias tensors.
pub(crate) fn build_bias<S, D, E, F>(biased: bool, dim: D, builder: F) -> Option<ArrayBase<S, E>>
where
    D: RemoveAxis<Smaller = E>,
    E: Dimension,
    F: Fn(E) -> ArrayBase<S, E>,
    S: RawData,
{
    if biased {
        Some(builder(bias_dim::<D, E>(dim)))
    } else {
        None
    }
}

pub(crate) fn bias_dim<D, E>(dim: D) -> E
where
    D: RemoveAxis<Smaller = E>,
    E: Dimension,
{
    if dim.ndim() == 1 {
        dim.remove_axis(Axis(0))
    } else {
        dim.remove_axis(Axis(1))
    }
}

/// A utilitarian function for checking if a type is [Biased]; returns false otherwise.
/// Compares the [TypeId](core::any::TypeId) of `K` to the [TypeId](core::any::TypeId) of [Biased].
pub fn is_biased<K: 'static>() -> bool {
    use core::any::TypeId;
    TypeId::of::<K>() == TypeId::of::<Biased>()
}
