/*
    Appellation: head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{_attention, Score};
use crate::params::QkvBase;
use concision::nn::Dropout;
use nd::linalg::Dot;
use nd::*;
use num::complex::ComplexFloat;

// #68
/// [AttentionHead] implements the scaled dot-product attention mechanism formally defined in
/// [Attention is all you need](https://arxiv.org/abs/1706.03762). The mechanism utilizes
/// [QkvBase] to store the query, key, and value tensors. More so, the attention head
/// has two optional features: a dropout and masking layer.
///
/// ### `dropout`
///
/// The [Dropout] layer is an optional, conditionally enabled layer (required the `rand` feature).
/// If enabled, the dropout layer is invoked after the softmax function is applied to the score.
/// The layer is used to prevent overfitting by randomly setting a fraction of the input
/// units to zero at each update during training time.
///
/// ### `mask`
///
/// After computing the dot-product of the query and key tensors, an optional mask may be applied to
/// the attention score. The mask is used to prevent the model from attending to certain parts of the
/// input sequence. For example, in the case of a language model, the mask may be used to prevent the
/// model from attending to the padding tokens.
pub struct AttentionHead<A = f64, D = Ix2, S = OwnedRepr<A>>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    #[cfg(feature = "rand")]
    pub(crate) dropout: Option<Dropout>,
    pub(crate) mask: Option<Array<bool, D>>,
    pub(crate) params: QkvBase<S, D>,
}

impl<A, S> AttentionHead<A, Ix2, S>
where
    S: RawData<Elem = A>,
{
    pub fn std(dm: usize, dk: usize) -> Self
    where
        A: Default,
        S: DataOwned,
    {
        Self::from_params(QkvBase::std(dk, dm))
    }
}

impl<A, S, D> AttentionHead<A, D, S>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Create a new instance with the specified shape and builder function.
    pub fn builder<Sh, F>(shape: Sh, builder: F) -> Self
    where
        F: Fn(D) -> ArrayBase<S, D>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_params(QkvBase::builder(shape, builder))
    }
    /// Create a new instance with the specified shape and value.
    pub fn from_elem<Sh>(shape: Sh, value: A) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone,
        S: DataOwned,
    {
        Self::from_params(QkvBase::from_elem(shape, value))
    }
    /// Create a new instance from the [QkvBase] parameters.
    pub fn from_params(params: QkvBase<S, D>) -> Self {
        Self {
            #[cfg(feature = "rand")]
            dropout: None,
            mask: None,
            params,
        }
    }
    /// Computes the [Score] using scaled dot-product attention.
    pub fn attention(&self) -> Score<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        S: Data,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        let (q, k, v) = self.qkv();
        _attention(q, k, v, self.mask(), self.dropout())
    }
    /// Returns an immutable reference to the, optional, mask.
    pub fn mask(&self) -> Option<&Array<bool, D>> {
        self.mask.as_ref()
    }
    /// Returns an immuable reference to the underlying parameters.
    pub const fn params(&self) -> &QkvBase<S, D> {
        &self.params
    }
    /// Returns a mutable reference to the underlying parameters.
    pub fn params_mut(&mut self) -> &mut QkvBase<S, D> {
        &mut self.params
    }
    /// Returns a three-tuple consisting of immputable references to the query, key, and value matrices respectively.
    pub fn qkv(&self) -> (&ArrayBase<S, D>, &ArrayBase<S, D>, &ArrayBase<S, D>) {
        self.params().qkv()
    }
    /// Consumes the head, returning a three-tuple consisting of mutable references to the query, key, and value matrices respectively.
    pub fn into_qkv(self) -> (ArrayBase<S, D>, ArrayBase<S, D>, ArrayBase<S, D>) {
        self.params.into_qkv()
    }
    /// Sets the dropout layer for the [AttentionHead]
    #[cfg(feature = "rand")]
    pub fn set_dropout(&mut self, dropout: Option<Dropout>) {
        self.dropout = dropout;
    }
    /// Sets the mask for the [AttentionHead]
    pub fn set_mask(&mut self, mask: Option<Array<bool, D>>) {
        self.mask = mask;
    }
    /// Configure the [AttentionHead] with a [DropoutLayer]
    #[cfg(feature = "rand")]
    pub fn with_dropout(self, dropout: Dropout) -> Self {
        Self {
            dropout: Some(dropout),
            ..self
        }
    }
    /// Consume and store a mask for the [AttentionHead]
    pub fn with_mask(self, mask: Array<bool, D>) -> Self {
        Self {
            mask: Some(mask),
            ..self
        }
    }

    ndbuilder!(default() where A: Default, S: DataOwned);
    ndbuilder!(ones() where A: Clone + num::One, S: DataOwned);
    ndbuilder!(zeros() where A: Clone + num::Zero, S: DataOwned);

    concision::dimensional!(params());
    concision::getters!(params::<[q, k, v]> => ArrayBase<S, D>);
}

#[cfg(feature = "rand")]
impl<A, S, D> AttentionHead<A, D, S>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Returns an immutable reference to the, optional, [dropout](DropoutLayer) layer.
    /// With the `rand` feature flag disabled, the dropout layer is
    /// unavailable and returns `None`.
    pub fn dropout(&self) -> Option<&Dropout> {
        self.dropout.as_ref()
    }
}

#[cfg(not(feature = "rand"))]
impl<A, S, D> AttentionHead<A, D, S>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Returns an immutable reference to the, optional, [dropout](DropoutLayer) layer.
    /// With the `rand` feature flag disabled, the dropout layer is
    /// unavailable and returns `None`.
    #[cfg(not(feature = "rand"))]
    pub fn dropout(&self) -> Option<&Dropout> {
        None
    }
}
