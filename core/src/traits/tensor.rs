/*
    Appellation: tensor <module>
    Contrib: @FL03
*/
use ndarray::{ArrayBase, DataOwned, Ix0};
use num::traits::{Num, NumAssign};

pub trait TensorScalar: Clone + Num + NumAssign {}

pub trait Tensor {
    type Elem: TensorScalar;
}

pub trait IsScalar {
    fn is_scalar(&self) -> bool;
}

pub trait FromScalar<T> {
    fn from_scalar(scalar: T) -> Self;
}

impl<S, D> IsScalar for ArrayBase<S, D>
where
    S: DataOwned,
    D: ndarray::Dimension + 'static,
{
    fn is_scalar(&self) -> bool {
        core::any::TypeId::of::<D>() == core::any::TypeId::of::<Ix0>()
    }
}
impl<A, S> FromScalar<A> for ArrayBase<S, Ix0>
where
    A: Clone,
    S: DataOwned<Elem = A>,
{
    fn from_scalar(scalar: A) -> Self {
        ArrayBase::from_elem((), scalar)
    }
}
