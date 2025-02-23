/*
   Appellation: pad <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#[doc(inline)]
pub use self::{error::*, mode::*, utils::*};

mod padding;

pub(crate) mod error;
pub(crate) mod mode;
pub(crate) mod utils;

///
pub trait Pad<T> {
    type Output;

    fn pad(&self, mode: PadMode<T>, pad: &[[usize; 2]]) -> Self::Output;
}

pub struct Padding<T> {
    pub(crate) action: PadAction,
    pub(crate) mode: PadMode<T>,
    pub(crate) pad: Vec<[usize; 2]>,
    pub(crate) padding: usize,
}

/*
 ************* Implementations *************
*/

use ndarray::{Array, ArrayBase, DataOwned, Dimension};
use num::traits::{FromPrimitive, Num};

impl<A, S, D> Pad<A> for ArrayBase<S, D>
where
    A: Copy + FromPrimitive + Num,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn pad(&self, mode: PadMode<A>, pad: &[[usize; 2]]) -> Self::Output {
        utils::pad(self, pad, mode).unwrap()
    }
}
