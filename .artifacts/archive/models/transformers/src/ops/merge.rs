/*
   Appellation: merge <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::_merge_dim;
use concision::NdResult;
use nd::{Array, ArrayBase, Data, Dimension, RemoveAxis};

pub trait DimMerge {
    type Output;

    fn merge(&self, tgt: usize) -> Self::Output;
}

// #67: Optimize the Merge trait
pub trait Merge {
    type Output;

    fn merge(&self) -> NdResult<Self::Output> {
        self.merge_along(0)
    }

    fn merge_along(&self, axis: usize) -> NdResult<Self::Output>;
}

/*
 ************* Implementations *************
*/
impl<D> DimMerge for D
where
    D: RemoveAxis,
    D::Smaller: Dimension,
    D::Larger: Dimension,
{
    type Output = D::Smaller;

    fn merge(&self, tgt: usize) -> Self::Output {
        _merge_dim(self, tgt)
    }
}

impl<A, S, D, E> Merge for ArrayBase<S, D>
where
    A: Clone,
    D: RemoveAxis<Smaller = E>,
    E: Dimension,
    S: Data<Elem = A>,
    ArrayBase<S, D>: Clone,
{
    type Output = Array<A, E>;

    fn merge(&self) -> NdResult<Self::Output> {
        let swap = if self.ndim() >= 3 { self.ndim() - 3 } else { 0 };
        self.merge_along(swap)
    }

    fn merge_along(&self, swap: usize) -> NdResult<Self::Output> {
        super::_merge(self, swap, swap + 1, super::ORDER)
    }
}
