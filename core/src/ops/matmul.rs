/*
    Appellation: matmul <module>
    Contrib: @FL03
*/

/// A trait denoting objects capable of matrix multiplication.
pub trait Matmul<Rhs = Self> {
    type Output;

    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

/*
 ************* Implementations *************
*/
// use ndarray::{ArrayBase, Data, Dimension, RawData};

// impl<A, S, D, X, Y> Matmul<X> for ArrayBase<S, D>
// where
//     A: ndarray::LinalgScalar,
//     D: Dimension,
//     S: Data<Elem = A>,
//     ArrayBase<S, D>: ndarray::linalg::Dot<X, Output = Y>,
// {
//     type Output = Y;

//     fn matmul(&self, rhs: &X) -> Self::Output {
//         self.dot(rhs)
//     }
// }

impl<T> Matmul<Vec<T>> for Vec<T>
where
    T: Copy + num::Num,
{
    type Output = T;

    fn matmul(&self, rhs: &Vec<T>) -> T {
        self.iter()
            .zip(rhs.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }
}

impl<T, const N: usize> Matmul<[T; N]> for [T; N]
where
    T: Copy + num::Num,
{
    type Output = T;

    fn matmul(&self, rhs: &[T; N]) -> T {
        self.iter()
            .zip(rhs.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }
}
