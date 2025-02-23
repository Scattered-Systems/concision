/*
    Appellation: attention <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention
//!
//! Attention allows a model to focus on specific parts of the input sequence.
//! Today, these mechanisms are found in several state-of-the-art models, such as
//! the Transformer model, primarily due to its capabilities in natural language
//! processing (NLP) domains
pub(crate) use self::_impl_methods::*;
pub use self::utils::*;
pub use self::{config::AttentionConfig, head::AttentionHead, score::Score};

pub(crate) mod config;
pub(crate) mod head;
pub(crate) mod score;

// #69: Multi-Head Attention implementation
pub mod multi;

pub(crate) mod prelude {
    pub use super::head::AttentionHead;
    pub use super::multi::prelude::*;
    pub use super::score::Score;
    pub use super::utils::*;
}

pub trait Attention {
    type Output;

    fn attention(&self) -> Self::Output;
}

pub(crate) mod utils {
    use super::Score;
    use concision::nn::Dropout;
    use nd::linalg::Dot;
    use nd::prelude::*;
    use num::complex::ComplexFloat;

    /// A functional implementation of the scaled dot-product attention mechanism;
    pub fn scaled_dot_product_attention<A, S, D>(
        q: &ArrayBase<S, D>,
        k: &ArrayBase<S, D>,
        v: &ArrayBase<S, D>,
        mask: Option<&Array<bool, D>>,
        dropout: Option<&Dropout>,
    ) -> Score<A, D>
    where
        A: ComplexFloat + nd::ScalarOperand,
        S: nd::Data<Elem = A>,
        D: Dimension,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        super::_attention(q, k, v, mask, dropout)
    }
}

mod _impl_methods {
    use super::Score;
    use concision::prelude::{Dropout, MaskFill, Softmax};
    use nd::linalg::Dot;
    use nd::prelude::*;
    use num::complex::ComplexFloat;

    pub(crate) fn _attention<A, S, D>(
        q: &ArrayBase<S, D>,
        k: &ArrayBase<S, D>,
        v: &ArrayBase<S, D>,
        mask: Option<&Array<bool, D>>,
        dropout: Option<&Dropout>,
    ) -> Score<A, D>
    where
        A: ComplexFloat + nd::ScalarOperand,
        S: nd::Data<Elem = A>,
        D: Dimension,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        let dk = scale::<A>(k.len_of(nd::Axis(1)));
        let mut z = q.dot(&k.t()) * dk;
        if let Some(mask) = mask {
            z = z.masked_fill(mask, A::zero());
        }
        z = z.softmax();
        #[cfg(feature = "rand")]
        if let Some(dropout) = dropout {
            z = concision::Predict::predict(dropout, &z).unwrap();
        }
        (z.dot(&v), z).into()
    }

    pub(crate) fn scale<A>(dk: usize) -> A
    where
        A: ComplexFloat,
    {
        A::from(dk).unwrap().sqrt().recip()
    }
}
