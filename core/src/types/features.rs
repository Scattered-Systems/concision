/*
    Appellation: shape <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Dimension, IntoDimension, Ix1, Ix2, ShapeError};

pub(crate) fn _from_dim<D>(dim: D) -> Result<Features, ShapeError>
where
    D: Dimension,
{
    if dim.ndim() == 1 {
        Ok(Features::new(dim[0], 1))
    } else if dim.ndim() >= 2 {
        Ok(Features::new(dim[1], dim[0]))
    } else {
        Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Features {
    inputs: usize,
    outputs: usize,
}

impl Features {
    /// Creates a new instance from the given number of input and output features
    pub fn new(inputs: usize, outputs: usize) -> Self {
        debug_assert_ne!(inputs, 0);
        debug_assert_ne!(outputs, 0);

        Self { inputs, outputs }
    }

    /// Attempts to build a new [Features] instance from the given dimension ([`D`](Dimension))
    pub fn from_dimension<D>(dim: D) -> Result<Self, ShapeError>
    where
        D: Dimension,
    {
        _from_dim(dim)
    }
    /// Builds a new instance from the given shape ([`Sh`](ShapeBuilder));
    /// Unlike [Features::from_dimension], this method requires the dimension (`D`) to
    /// additionally implement the [RemoveAxis] trait
    pub fn from_shape<D, Sh>(shape: Sh) -> Self
    where
        D: ndarray::RemoveAxis,
        Sh: ndarray::ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape_with_order().raw_dim().clone();
        _from_dim(dim).unwrap()
    }
    /// Creates a new instance given the model size (`inputs`, `d_model`) and total number of nodes within the network (`size`, `network`, `d_network`)
    pub fn from_network(model: usize, network: usize) -> Self {
        let outputs = network / model;
        Self::new(model, outputs)
    }

    pub const fn as_array(&self) -> [usize; 2] {
        [self.outputs(), self.inputs()]
    }
    /// Creates a new two-tuple instance from the given dimensions;
    pub const fn as_tuple(&self) -> (usize, usize) {
        (self.outputs(), self.inputs())
    }
    /// Checks to see if the given dimension is compatible with the features
    pub fn check_dim<D>(&self, dim: D) -> bool
    where
        D: Dimension,
    {
        if dim.ndim() == 1 {
            self.inputs() == dim[0]
        } else if dim.ndim() >= 2 {
            self.outputs() == dim[0] && self.inputs() == dim[1]
        } else {
            false
        }
    }
    /// Forwards the [into_pattern](ndarray::Dimension::into_pattern) method from the [Dimension] trait
    #[inline]
    pub fn into_pattern(self) -> (usize, usize) {
        self.into_dimension().into_pattern()
    }
    /// An aliased function that returns the number of input features
    pub const fn d_model(&self) -> usize {
        self.inputs()
    }

    /// Returns the number of input features
    pub const fn inputs(&self) -> usize {
        self.inputs
    }
    /// a utilitarian function that checks if the output features is equal to 1
    pub fn is_unit(&self) -> bool {
        self.outputs() == 1
    }
    /// Returns the number of output features
    pub const fn outputs(&self) -> usize {
        self.outputs
    }
    /// Computes the total number of nodes in the network
    pub fn size(&self) -> usize {
        self.inputs() * self.outputs()
    }
    #[doc(hidden)]
    pub fn uniform_scale(&self) -> f64 {
        (self.inputs as f64).recip().sqrt()
    }
}

impl IntoDimension for Features {
    type Dim = Ix2;

    fn into_dimension(self) -> Self::Dim {
        (self.outputs, self.inputs).into_dimension()
    }
}

impl From<Ix1> for Features {
    fn from(dim: Ix1) -> Self {
        Self::new(1, dim[0])
    }
}

impl From<Ix2> for Features {
    fn from(dim: Ix2) -> Self {
        Self::new(dim[1], dim[0])
    }
}

impl From<Features> for Ix2 {
    fn from(features: Features) -> Self {
        features.into_dimension()
    }
}

impl<U> PartialEq<U> for Features
where
    [usize; 2]: PartialEq<U>,
{
    fn eq(&self, other: &U) -> bool {
        self.as_array() == *other
    }
}
