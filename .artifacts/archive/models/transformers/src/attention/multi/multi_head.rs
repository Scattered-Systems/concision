/*
    Appellation: multi_head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{AttentionHead, attention::AttentionConfig};
use linear::{Biased, Linear};
use nd::prelude::*;
use nd::{DataOwned, OwnedRepr, RawData};

// #69
pub struct MultiHeadAttention<A = f64, D = Ix2, S = OwnedRepr<A>>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub(crate) config: AttentionConfig,
    pub(crate) head: AttentionHead<A, D, S>,
    pub(crate) linears: Vec<Linear<A, Biased, D, S>>,
}

impl<A, S, D> MultiHeadAttention<A, D, S>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Returns an immutable reference to the [AttentionConfig]
    pub const fn config(&self) -> &AttentionConfig {
        &self.config
    }
    /// Returns an immutable reference to the [AttentionHead]
    pub const fn head(&self) -> &AttentionHead<A, D, S> {
        &self.head
    }
    /// Returns a mutable reference to the [AttentionHead]
    pub fn head_mut(&mut self) -> &mut AttentionHead<A, D, S> {
        &mut self.head
    }
    /// Returns an immutable slice containing the [Linear] layers
    pub fn linears(&self) -> &[Linear<A, Biased, D, S>] {
        &self.linears
    }
}

impl<A, S> MultiHeadAttention<A, Ix2, S>
where
    S: RawData<Elem = A>,
{
    pub fn std(d_model: usize, heads: usize) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
    {
        let config = AttentionConfig::new(d_model, heads);
        let linears = (0..4)
            .map(|_| Linear::from_features(d_model, d_model))
            .collect();
        Self {
            config,
            head: AttentionHead::std(d_model, config.dk()),
            linears,
        }
    }
}

impl<A, S, D> Default for MultiHeadAttention<A, D, S>
where
    A: Default,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    fn default() -> Self {
        Self {
            config: AttentionConfig::default(),
            head: Default::default(),
            linears: Vec::new(),
        }
    }
}
