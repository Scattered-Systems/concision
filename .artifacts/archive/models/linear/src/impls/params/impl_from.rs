/*
    Appellation: impl_from <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Biased, Features, NodeBase, Pair, ParamsBase, Unbiased};
#[cfg(all(feature = "alloc", no_std))]
use alloc::vec;
use core::marker::PhantomData;
use nd::prelude::*;
use nd::{Data, DataMut, DataOwned, OwnedRepr, RawData, RemoveAxis};
#[cfg(feature = "std")]
use std::vec;

impl<A, S, D, E> IntoIterator for ParamsBase<S, D, Biased>
where
    A: Clone,
    D: Dimension<Smaller = E> + RemoveAxis,
    S: Data<Elem = A>,
    E: RemoveAxis,
{
    type Item = (Array<A, E>, Array<A, E::Smaller>);
    type IntoIter = vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let axis = Axis(0);
        self.weights()
            .axis_iter(axis)
            .zip(self.bias().axis_iter(axis))
            .map(|(w, b)| (w.to_owned(), b.to_owned()))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<A, S, D, E> IntoIterator for ParamsBase<S, D, Unbiased>
where
    A: Clone,
    D: Dimension<Smaller = E> + RemoveAxis,
    S: Data<Elem = A>,
    E: RemoveAxis,
{
    type Item = Array<A, E>;
    type IntoIter = vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.weights()
            .axis_iter(Axis(0))
            .map(|w| w.to_owned())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<A, S> FromIterator<(Array1<A>, Option<Array0<A>>)> for ParamsBase<S, Ix2>
where
    A: Clone + Default,
    S: DataOwned<Elem = A> + DataMut,
{
    fn from_iter<I: IntoIterator<Item = (Array1<A>, Option<Array0<A>>)>>(nodes: I) -> Self {
        let nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut iter = nodes.iter();
        let node = iter.next().unwrap();
        let shape = Features::new(node.0.len(), nodes.len());
        let mut params = ParamsBase::new(shape);
        params.set_node(0, node.clone());
        for (i, node) in iter.into_iter().enumerate() {
            params.set_node(i + 1, node.clone());
        }
        params
    }
}

macro_rules! impl_from {
    ($($bias:ty),*) => {
        $(impl_from!(@impl $bias);)*

    };
    (@impl $b:ty) => {

    };
}

impl_from!(ArrayBase<S, D::Smaller>);

impl<A> From<(Array1<A>, A)> for ParamsBase<OwnedRepr<A>, Ix1, Biased>
where
    A: Clone,
{
    fn from((weights, bias): (Array1<A>, A)) -> Self {
        let bias = ArrayBase::from_elem((), bias);
        Self {
            bias: Some(bias),
            weight: weights,
            _mode: PhantomData,
        }
    }
}
impl<A, K> From<(Array1<A>, Option<A>)> for ParamsBase<OwnedRepr<A>, Ix1, K>
where
    A: Clone,
{
    fn from((weights, bias): (Array1<A>, Option<A>)) -> Self {
        Self {
            bias: bias.map(|b| ArrayBase::from_elem((), b)),
            weight: weights,
            _mode: PhantomData,
        }
    }
}

impl<A, S, D, K> From<NodeBase<S, D, D::Smaller>> for ParamsBase<S, D, K>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    fn from((weights, bias): NodeBase<S, D, D::Smaller>) -> Self {
        Self {
            bias,
            weight: weights,
            _mode: PhantomData::<K>,
        }
    }
}

impl<A, S, D> From<Pair<ArrayBase<S, D>, ArrayBase<S, D::Smaller>>> for ParamsBase<S, D, Biased>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    fn from((weights, bias): Pair<ArrayBase<S, D>, ArrayBase<S, D::Smaller>>) -> Self {
        Self {
            bias: Some(bias),
            weight: weights,
            _mode: PhantomData::<Biased>,
        }
    }
}
