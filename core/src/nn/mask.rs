/*
    Appellation: mask <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::iter::{Iter, IterMut};
use ndarray::{ArrayBase, Data, DataMut, Dimension, Ix2, OwnedRepr, RawData, RawDataClone};

pub struct Mask<S = OwnedRepr<bool>, D = Ix2>(ArrayBase<S, D>)
where
    D: Dimension,
    S: RawData<Elem = bool>;

impl<S, D> Mask<S, D>
where
    D: Dimension,
    S: RawData<Elem = bool>,
{
    pub fn from_arr(data: ArrayBase<S, D>) -> Self {
        Self(data)
    }

    pub fn apply<A, T, F>(&mut self, data: &ArrayBase<T, D>, fill: A) -> ArrayBase<T, D>
    where
        A: Clone,
        S: Data,
        T: DataMut<Elem = A> + RawDataClone,
    {
        let mut res = data.clone();
        res.zip_mut_with(self.as_mut(), |x, &m| {
            if m {
                *x = fill.clone();
            }
        });
        res
    }

    pub fn mask_inplace<'a, A, T, F>(
        &mut self,
        data: &'a mut ArrayBase<T, D>,
        fill: A,
    ) -> &'a mut ArrayBase<T, D>
    where
        A: Clone,
        S: Data,
        T: DataMut<Elem = A>,
    {
        data.zip_mut_with(&mut self.0, |x, &m| {
            if m {
                *x = fill.clone();
            }
        });
        data
    }

    pub fn as_slice(&self) -> &[bool]
    where
        S: Data,
    {
        self.get().as_slice().unwrap()
    }

    pub fn as_mut_slice(&mut self) -> &mut [bool]
    where
        S: DataMut,
    {
        self.get_mut().as_slice_mut().unwrap()
    }

    pub fn dim(&self) -> D::Pattern {
        self.get().dim()
    }

    pub fn iter(&self) -> Iter<'_, bool, D>
    where
        S: Data,
    {
        self.get().iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, bool, D>
    where
        S: DataMut,
    {
        self.get_mut().iter_mut()
    }

    pub fn get(&self) -> &ArrayBase<S, D> {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.0
    }

    pub fn into_inner(self) -> ArrayBase<S, D> {
        self.0
    }

    pub fn ndim(&self) -> usize {
        self.get().ndim()
    }

    pub fn raw_dim(&self) -> D {
        self.get().raw_dim()
    }

    pub fn set(&mut self, data: ArrayBase<S, D>) {
        self.0 = data;
    }

    pub fn shape(&self) -> D {
        self.get().raw_dim()
    }
}

/*
 ************* Implementations *************
*/
mod impls {
    use super::Mask;
    use core::borrow::{Borrow, BorrowMut};
    use core::ops::{Deref, DerefMut, Index, IndexMut};
    use ndarray::{ArrayBase, Data, DataMut, Dimension, NdIndex, RawData};

    impl<S, D> AsRef<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn as_ref(&self) -> &ArrayBase<S, D> {
            &self.0
        }
    }

    impl<S, D> AsMut<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn as_mut(&mut self) -> &mut ArrayBase<S, D> {
            &mut self.0
        }
    }

    impl<S, D> Borrow<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn borrow(&self) -> &ArrayBase<S, D> {
            &self.0
        }
    }

    impl<S, D> BorrowMut<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn borrow_mut(&mut self) -> &mut ArrayBase<S, D> {
            &mut self.0
        }
    }

    impl<S, D> Deref for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        type Target = ArrayBase<S, D>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<S, D> DerefMut for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl<S, D, I> Index<I> for Mask<S, D>
    where
        D: Dimension,
        I: NdIndex<D>,
        S: Data<Elem = bool>,
    {
        type Output = <ArrayBase<S, D> as Index<I>>::Output;

        fn index(&self, index: I) -> &Self::Output {
            &self.0[index]
        }
    }

    impl<S, D, I> IndexMut<I> for Mask<S, D>
    where
        D: Dimension,
        I: NdIndex<D>,
        S: DataMut<Elem = bool>,
    {
        fn index_mut(&mut self, index: I) -> &mut Self::Output {
            &mut self.0[index]
        }
    }
}

mod impl_from {
    use super::Mask;
    use ndarray::{ArrayBase, Dimension, RawData};

    impl<S, D> From<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn from(mask: ArrayBase<S, D>) -> Self {
            Mask(mask)
        }
    }

    impl<S, D> From<Mask<S, D>> for ArrayBase<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn from(mask: Mask<S, D>) -> Self {
            mask.0
        }
    }
}
