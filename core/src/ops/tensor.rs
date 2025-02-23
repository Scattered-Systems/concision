/*
    Appellation: ops <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::linalg::Dot;
use ndarray::*;
use num::traits::{Num, NumAssign};

pub trait Affine<X, Y = X> {
    type Output;

    fn affine(&self, mul: X, add: Y) -> Self::Output;
}

pub trait Inverse {
    type Output;

    fn inverse(&self) -> Self::Output;
}

pub trait Matpow<Rhs = Self> {
    type Output;

    fn pow(&self, rhs: Rhs) -> Self::Output;
}

/*
 ********* Implementations *********
*/
impl<A, D> Affine<A> for Array<A, D>
where
    A: LinalgScalar + ScalarOperand,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn affine(&self, mul: A, add: A) -> Self::Output {
        self * mul + add
    }
}

// #[cfg(feature = "blas")]
impl<T> Inverse for Array<T, Ix2>
where
    T: Copy + NumAssign + ScalarOperand,
{
    type Output = Option<Self>;
    fn inverse(&self) -> Self::Output {
        crate::inverse(self)
    }
}

impl<A, S> Matpow<i32> for ArrayBase<S, ndarray::Ix2>
where
    A: Copy + Num + 'static,
    S: Data<Elem = A>,
    ArrayBase<S, Ix2>: Clone + Dot<ArrayBase<S, Ix2>, Output = Array<A, Ix2>>,
{
    type Output = Array<A, Ix2>;

    fn pow(&self, rhs: i32) -> Self::Output {
        if !self.is_square() {
            panic!("Matrix must be square to be raised to a power");
        }
        let mut res = Array::eye(self.shape()[0]);
        for _ in 0..rhs {
            res = res.dot(self);
        }
        res
    }
}
