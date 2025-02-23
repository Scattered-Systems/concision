/*
    Appellation: reshape <module> [traits::arr]
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::*;
use ndarray::{RawData, RawDataClone};

pub trait Unsqueeze {
    type Output;

    fn unsqueeze(self, axis: usize) -> Self::Output;
}

/*
 ************* Implementations *************
*/

impl<A, S, D> Unsqueeze for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Output = ArrayBase<S, D::Larger>;

    fn unsqueeze(self, axis: usize) -> Self::Output {
        self.insert_axis(Axis(axis))
    }
}

impl<'a, A, S, D> Unsqueeze for &'a ArrayBase<S, D>
where
    D: Dimension,
    S: RawDataClone<Elem = A>,
{
    type Output = ArrayBase<S, D::Larger>;

    fn unsqueeze(self, axis: usize) -> Self::Output {
        self.clone().insert_axis(Axis(axis))
    }
}
