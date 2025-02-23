/*
    Appellation: layer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::attention::multi::MultiHeadAttention;
use crate::model::ffn::FeedForwardNetwork;
use linear::Biased;
use nd::prelude::*;

pub struct EncoderLayer<A = f64, K = Biased, D = Ix2>
where
    D: Dimension,
{
    pub(crate) attention: MultiHeadAttention<A, D>,
    pub(crate) ffn: FeedForwardNetwork<A, K, D>,
}

impl<A, D, K> EncoderLayer<A, K, D>
where
    D: Dimension,
{
    pub fn new(attention: MultiHeadAttention<A, D>, ffn: FeedForwardNetwork<A, K, D>) -> Self {
        Self { attention, ffn }
    }
    /// Returns an immutable reference to the multi-head, self-attention layer.
    pub fn attention(&self) -> &MultiHeadAttention<A, D> {
        &self.attention
    }
    /// Returns a mutable reference to the multi-head, self-attention layer.
    pub fn attention_mut(&mut self) -> &mut MultiHeadAttention<A, D> {
        &mut self.attention
    }
    /// Returns an immutable reference to the feed-forward network layer.
    pub fn ffn(&self) -> &FeedForwardNetwork<A, K, D> {
        &self.ffn
    }
    /// Returns a mutable reference to the feed-forward network layer.
    pub fn ffn_mut(&mut self) -> &mut FeedForwardNetwork<A, K, D> {
        &mut self.ffn
    }
}
