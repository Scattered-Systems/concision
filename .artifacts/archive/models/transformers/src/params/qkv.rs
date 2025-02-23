/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::attention::{_attention, Score};
use concision::{dimensional, getters};
use nd::linalg::Dot;
use nd::*;
use num::complex::ComplexFloat;
use num::traits::{One, Zero};

/// [QkvBase] is a container for the query, key, and value arrays used in the
/// attention mechanism of the transformer model.
pub struct QkvBase<S = OwnedRepr<f64>, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) q: ArrayBase<S, D>,
    pub(crate) k: ArrayBase<S, D>,
    pub(crate) v: ArrayBase<S, D>,
}

impl<A, S, D> QkvBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn builder<Sh, F>(shape: Sh, builder: F) -> Self
    where
        F: Fn(D) -> ArrayBase<S, D>,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape_with_order().raw_dim().clone();
        Self {
            q: builder(dim.clone()),
            k: builder(dim.clone()),
            v: builder(dim),
        }
    }

    pub fn from_elem<Sh>(shape: Sh, value: A) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone,
        S: DataOwned,
    {
        let dim = shape.into_shape_with_order().raw_dim().clone();
        Self {
            q: ArrayBase::from_elem(dim.clone(), value.clone()),
            k: ArrayBase::from_elem(dim.clone(), value.clone()),
            v: ArrayBase::from_elem(dim, value),
        }
    }

    pub fn as_qkv(&self) -> (ArrayView<A, D>, ArrayView<A, D>, ArrayView<A, D>)
    where
        S: Data,
    {
        (self.q.view(), self.k.view(), self.v.view())
    }

    /// Consumes the store and returns a three-tuple consisting of the query, key, and value arrays respectively.
    pub fn into_qkv(self) -> (ArrayBase<S, D>, ArrayBase<S, D>, ArrayBase<S, D>) {
        (self.q, self.k, self.v)
    }

    pub fn qkv(&self) -> (&ArrayBase<S, D>, &ArrayBase<S, D>, &ArrayBase<S, D>) {
        (&self.q, &self.k, &self.v)
    }

    ndbuilder!(new::default() where A: Default, S: DataOwned);
    ndbuilder!(ones() where A: Clone + One, S: DataOwned);
    ndbuilder!(zeros() where A: Clone + Zero, S: DataOwned);

    getters!(q, k, v => ArrayBase<S, D>);

    dimensional!(q());

    qkv_view!(into_owned::<OwnedRepr>(self) where A: Clone, S: Data);
    qkv_view!(to_owned::<OwnedRepr>(&self) where A: Clone, S: Data);

    qkv_view!(into_shared::<OwnedArcRepr>(self) where A: Clone, S: DataOwned);
    qkv_view!(to_shared::<OwnedArcRepr>(&self) where A: Clone, S: DataShared);

    qkv_view!(view::<'a, ViewRepr>(&self) where S: Data);
    qkv_view!(view_mut::<'a, ViewRepr>(&mut self) where S: DataMut);
}

impl<A, S> QkvBase<S, Ix2>
where
    S: RawData<Elem = A>,
{
    pub fn std(dk: usize, d_model: usize) -> Self
    where
        A: Default,
        S: DataOwned,
    {
        Self::new((dk, d_model))
    }
}

#[cfg(not(feature = "rand"))]
impl<A, S, D> QkvBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
    A: Clone,
{
    /// Computes the [Score] using scaled dot-product attention.
    pub fn attention(&self, dropout: Option<f64>, mask: Option<&Array<bool, D>>) -> Score<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        S: Data,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        let (q, k, v) = self.qkv();
        _attention(q, k, v, mask, None)
    }
}

#[cfg(feature = "rand")]
impl<A, S, D> QkvBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
    A: Clone,
{
    /// Computes the [Score] using scaled dot-product attention.
    pub fn attention(&self, dropout: Option<f64>, mask: Option<&Array<bool, D>>) -> Score<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        S: Data,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        use concision::nn::Dropout;

        let dropout = dropout.map(Dropout::new);
        let (q, k, v) = self.qkv();
        _attention(q, k, v, mask, dropout.as_ref())
    }
}
