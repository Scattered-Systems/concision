/*
    Appellation: impl_linalg <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::{Qkv, QkvBase};
use concision::Matmul;
use nd::linalg::Dot;
use nd::*;

impl<A, S, T, D, E, F> Matmul<QkvBase<T, E>> for QkvBase<S, D>
where
    A: LinalgScalar,
    D: Dimension,
    E: Dimension,
    F: Dimension,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
    ArrayBase<S, D>: Dot<ArrayBase<T, E>, Output = Array<A, F>>,
{
    type Output = Qkv<A, F>;

    fn matmul(&self, rhs: &QkvBase<T, E>) -> Self::Output {
        QkvBase {
            q: self.q().dot(rhs.q()),
            k: self.k().dot(rhs.k()),
            v: self.v().dot(rhs.v()),
        }
    }
}

impl<A, S, T, D, E, F> Matmul<ArrayBase<T, E>> for QkvBase<S, D>
where
    A: LinalgScalar,
    D: Dimension,
    E: Dimension,
    F: Dimension,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
    ArrayBase<S, D>: Dot<ArrayBase<T, E>, Output = Array<A, F>>,
{
    type Output = Qkv<A, F>;

    fn matmul(&self, rhs: &ArrayBase<T, E>) -> Self::Output {
        QkvBase {
            q: self.q().dot(rhs),
            k: self.k().dot(rhs),
            v: self.v().dot(rhs),
        }
    }
}
