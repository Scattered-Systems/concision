/*
    Appellation: model <module>
    Contrib: @FL03
*/
use ndarray::{ArrayBase, Dimension, RawData};

pub trait Parameters<S, D = ndarray::Ix1>
where
    D: Dimension,
    S: RawData,
{
    fn bias(&self) -> ArrayBase<S, D::Smaller>;
    fn weights(&self) -> ArrayBase<S, D>;
}
