/*
    Appellation: impl_params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::QkvBase;
use nd::prelude::*;
use nd::{Data, DataOwned, RawDataClone};

pub(crate) type ThreeTuple<A, B = A, C = B> = (A, B, C);

impl<A, S, D> Clone for QkvBase<S, D>
where
    D: Dimension,
    S: RawDataClone<Elem = A>,
{
    fn clone(&self) -> Self {
        Self {
            q: self.q.clone(),
            k: self.k.clone(),
            v: self.v.clone(),
        }
    }
}

impl<A, S, D> Copy for QkvBase<S, D>
where
    D: Copy + Dimension,
    S: Copy + RawDataClone<Elem = A>,
{
}

impl<A, S, D> Default for QkvBase<S, D>
where
    A: Default,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    fn default() -> Self {
        Self {
            q: Default::default(),
            k: Default::default(),
            v: Default::default(),
        }
    }
}

impl<A, S, D> PartialEq for QkvBase<S, D>
where
    A: PartialEq,
    D: Dimension,
    S: Data<Elem = A>,
{
    fn eq(&self, other: &Self) -> bool {
        self.q() == other.q() && self.k() == other.k() && self.v() == other.v()
    }
}

impl<A, B, S, D, S2, D2> PartialEq<ArrayBase<S2, D2>> for QkvBase<S, D>
where
    A: PartialEq,
    B: PartialEq,
    D: Dimension,
    S: Data<Elem = A>,
    S2: Data<Elem = B>,
    D2: Dimension,
    ArrayBase<S, D>: PartialEq<ArrayBase<S2, D2>>,
{
    fn eq(&self, other: &ArrayBase<S2, D2>) -> bool {
        self.q() == other && self.k() == other && self.v() == other
    }
}

impl<A, B, S, D, S2, D2> PartialEq<ThreeTuple<ArrayBase<S2, D2>>> for QkvBase<S, D>
where
    A: PartialEq,
    B: PartialEq,
    D: Dimension,
    S: Data<Elem = A>,
    S2: Data<Elem = B>,
    D2: Dimension,
    ArrayBase<S, D>: PartialEq<ArrayBase<S2, D2>>,
{
    fn eq(&self, (q, k, v): &ThreeTuple<ArrayBase<S2, D2>>) -> bool {
        self.q() == q && self.k() == k && self.v() == v
    }
}
