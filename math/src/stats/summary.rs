/*
    Appellation: summary <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::Root;
use core::iter::{Product, Sum};
use ndarray::{ArrayBase, Data, Dimension};
use num::traits::{FromPrimitive, Num, NumOps, Pow};

/// This trait describes the fundamental methods of summary statistics.
/// These include the mean, standard deviation, variance, and more.
pub trait SummaryStatistics
where
    Self::Item: FromPrimitive,
    Self::Output: NumOps<Self::Item, Self::Output>,
{
    type Item;
    type Output;

    fn elems(&self) -> Self::Item {
        Self::Item::from_usize(self.len()).unwrap()
    }

    fn len(&self) -> usize;

    fn mean(&self) -> Self::Output {
        self.sum() / self.elems()
    }

    fn product(&self) -> Self::Output;

    fn sum(&self) -> Self::Output;

    fn std(&self) -> Self::Output;

    fn var(&self) -> Self::Output;
}

/*
 ************* Implementations *************
*/
impl<'a, T, I> SummaryStatistics for &'a I
where
    I: Clone + ExactSizeIterator<Item = T>,
    T: Copy + FromPrimitive + Num + Pow<i32, Output = T> + Product + Root<Output = T> + Sum,
{
    type Item = T;
    type Output = T;

    fn len(&self) -> usize {
        ExactSizeIterator::len(*self)
    }

    fn product(&self) -> Self::Output {
        (*self).clone().product()
    }

    fn sum(&self) -> Self::Output {
        (*self).clone().sum()
    }

    fn std(&self) -> Self::Output {
        let mean = self.mean();
        let sum = (*self).clone().map(|x| (x - mean).pow(2)).sum::<T>();
        (sum / self.elems()).sqrt()
    }

    fn var(&self) -> Self::Output {
        let mean = self.mean();
        let sum = (*self).clone().map(|x| (x - mean).pow(2)).sum::<T>();
        sum / self.elems()
    }
}

macro_rules! impl_summary {
    ($($T:ty),* $(,)?) => {
        $(
            impl_summary!(@impl $T);
        )*
    };
    (@impl $T:ty) => {

        impl<T> SummaryStatistics for $T
        where
            T: Copy + FromPrimitive + Num + Pow<i32, Output = T> + Product + Root<Output = T> + Sum,
        {
            type Item = T;
            type Output = T;

            fn len(&self) -> usize {
                self.len()
            }

            fn product(&self) -> Self::Output {
                self.iter().copied().product::<T>()
            }

            fn sum(&self) -> Self::Output {
                self.iter().copied().sum::<T>()
            }

            fn std(&self) -> Self::Output {
                let mean = self.mean();
                let sum = self.iter().copied().map(|x| (x - mean).pow(2)).sum::<T>();
                (sum / self.elems()).sqrt()
            }

            fn var(&self) -> Self::Output {
                let mean = self.mean();
                let sum = self.iter().copied().map(|x| (x - mean).pow(2)).sum::<T>();
                sum / self.elems()
            }
        }
    };
}

impl_summary!(Vec<T>, [T]);

impl<A, S, D> SummaryStatistics for ArrayBase<S, D>
where
    A: Copy + FromPrimitive + Num + Pow<i32, Output = A> + Product + Root<Output = A> + Sum,
    D: Dimension,
    S: Data<Elem = A>,
    for<'a> &'a A: Product,
{
    type Item = A;
    type Output = A;

    fn len(&self) -> usize {
        self.len()
    }

    fn product(&self) -> Self::Output {
        self.iter().copied().product::<A>()
    }

    fn sum(&self) -> Self::Output {
        self.iter().copied().sum::<A>()
    }

    fn std(&self) -> Self::Output {
        let mean = self.mean().unwrap_or_else(A::zero);
        let sum = self.iter().copied().map(|x| (x - mean).pow(2)).sum::<A>();
        (sum / self.elems()).sqrt()
    }

    fn var(&self) -> Self::Output {
        let mean = self.mean().unwrap_or_else(A::zero);
        let sum = self.iter().copied().map(|x| (x - mean).pow(2)).sum::<A>();
        sum / self.elems()
    }
}
