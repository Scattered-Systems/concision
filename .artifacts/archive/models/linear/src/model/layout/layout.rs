/*
   Appellation: layout <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::model::layout::{Features, features};
use core::borrow::Borrow;
use nd::{Dimension, RemoveAxis, ShapeBuilder, ShapeError};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]

pub struct Layout<D = nd::Ix2> {
    pub(crate) dim: D,
    pub(crate) features: Features,
}

impl<D> Layout<D>
where
    D: Dimension,
{
    pub fn new(dim: D) -> Self
    where
        D: RemoveAxis,
    {
        Self::from_dim(dim).expect("Invalid dimension")
    }

    pub fn from_dim(dim: D) -> Result<Self, ShapeError> {
        let features = features(dim.clone())?;
        Ok(Self { dim, features })
    }

    pub fn from_shape<Sh>(shape: Sh) -> Self
    where
        D: RemoveAxis,
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape_with_order();
        let dim = shape.raw_dim().clone();
        let features = Features::from_shape(shape);
        Self { dim, features }
    }

    pub fn with_shape<E, Sh>(self, shape: Sh) -> Layout<E>
    where
        E: RemoveAxis,
        Sh: ShapeBuilder<Dim = E>,
    {
        let shape = shape.into_shape_with_order();
        let dim = shape.raw_dim().clone();
        let features = Features::from_shape(dim.clone());
        Layout { dim, features }
    }

    pub fn as_slice(&self) -> &[usize] {
        self.dim.slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        self.dim.slice_mut()
    }

    pub fn dim(&self) -> D {
        self.dim.clone()
    }

    pub fn features(&self) -> Features {
        self.features
    }

    pub fn into_dimensionality<E>(self, dim: E) -> Result<Layout<E>, ShapeError>
    where
        E: Dimension,
    {
        Layout::from_dim(dim)
    }

    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    pub fn pattern(&self) -> D::Pattern
    where
        D: Copy,
    {
        self.dim.into_pattern()
    }
}

impl<D> Borrow<D> for Layout<D>
where
    D: Dimension,
{
    fn borrow(&self) -> &D {
        &self.dim
    }
}

impl<D> Borrow<Features> for Layout<D>
where
    D: Dimension,
{
    fn borrow(&self) -> &Features {
        &self.features
    }
}

impl<D> PartialEq<D> for Layout<D>
where
    D: Dimension,
{
    fn eq(&self, other: &D) -> bool {
        self.dim.eq(other)
    }
}

impl<D> PartialEq<Features> for Layout<D>
where
    D: Dimension,
{
    fn eq(&self, other: &Features) -> bool {
        self.features.eq(other)
    }
}
