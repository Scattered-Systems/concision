/*
   Appellation: adjust <traits>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Axis, RemoveAxis};

/// Decrement generally describes an object capable of _decrementing_ itself;
///
/// Here, it is used on a [Dimension](ndarray::Dimension) enabling it to
/// remove and return an axis from itself.
pub trait Decrement {
    type Output;

    fn dec(&self) -> Self::Output;
}

pub trait Increment {
    type Output;

    fn inc(&self) -> Self::Output;
}

/*
 ******** implementations ********
*/
impl<D> Decrement for D
where
    D: RemoveAxis,
{
    type Output = D::Smaller;

    fn dec(&self) -> Self::Output {
        self.remove_axis(Axis(self.ndim() - 1))
    }
}
