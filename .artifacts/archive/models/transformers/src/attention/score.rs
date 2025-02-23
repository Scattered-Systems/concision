/*
    Appellation: score <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use core::fmt;
use nd::{Array, Dimension};

/// [Score] is a created as a result of invoking an attention mechanism;
///
/// - attention: the actual result; returns the dot product of the score with the value tensor
/// - score: the attention score tensor
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct Score<A, D>
where
    D: Dimension,
{
    pub(crate) attention: Array<A, D>,
    pub(crate) score: Array<A, D>,
}

impl<A, D> Score<A, D>
where
    D: Dimension,
{
    pub(crate) fn new(attention: Array<A, D>, score: Array<A, D>) -> Self {
        Self { attention, score }
    }
    /// Consumes the instance and returns the attention tensor.
    pub fn into_attention(self) -> Array<A, D> {
        self.attention
    }
    /// Consumes the container and returns the score tensor.
    pub fn into_score(self) -> Array<A, D> {
        self.score
    }
    /// Retrieve the attention tensor.
    pub fn attention(&self) -> &Array<A, D> {
        &self.attention
    }
    /// Retrieve the score tensor
    pub fn score(&self) -> &Array<A, D> {
        &self.score
    }
}

impl<A, D> Copy for Score<A, D>
where
    A: Copy,
    D: Copy + Dimension,
    Array<A, D>: Copy,
{
}

impl<A, D> fmt::Debug for Score<A, D>
where
    A: fmt::Debug,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Score")
            .field("attention", &self.attention)
            .field("score", &self.score)
            .finish()
    }
}

impl<A, D> fmt::Display for Score<A, D>
where
    A: fmt::Display,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.attention, self.score)
    }
}

impl<A, D> From<(Array<A, D>, Array<A, D>)> for Score<A, D>
where
    D: Dimension,
{
    fn from((attention, score): (Array<A, D>, Array<A, D>)) -> Self {
        Self::new(attention, score)
    }
}

impl<A, D> From<Score<A, D>> for (Array<A, D>, Array<A, D>)
where
    D: Dimension,
{
    fn from(score: Score<A, D>) -> Self {
        (score.attention, score.score)
    }
}
