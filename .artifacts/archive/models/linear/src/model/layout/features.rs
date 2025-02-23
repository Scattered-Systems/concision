/*
   Appellation: features <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::prelude::{Dimension, Ix2, ShapeBuilder};
use nd::{ErrorKind, IntoDimension, RemoveAxis, ShapeError};

pub(crate) fn features<D>(dim: D) -> Result<Features, ShapeError>
where
    D: Dimension,
{
    if dim.ndim() == 1 {
        Ok(Features::new(1, dim[0]))
    } else if dim.ndim() >= 2 {
        Ok(Features::new(dim[0], dim[1]))
    } else {
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Features {
    pub dmodel: usize,  // inputs
    pub outputs: usize, // outputs
}

impl Features {
    pub fn new(outputs: usize, dmodel: usize) -> Self {
        Self { dmodel, outputs }
    }

    pub fn from_dim<D>(dim: D) -> Self
    where
        D: RemoveAxis,
    {
        features(dim).unwrap()
    }

    pub fn from_shape<D, Sh>(shape: Sh) -> Self
    where
        D: RemoveAxis,
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape_with_order();
        let dim = shape.raw_dim().clone();
        features(dim).unwrap()
    }

    pub fn check_dim<D>(&self, dim: D) -> bool
    where
        D: Dimension,
    {
        if dim.ndim() == 1 {
            self.dmodel == dim[0]
        } else if dim.ndim() >= 2 {
            self.outputs == dim[0] && self.dmodel == dim[1]
        } else {
            false
        }
    }

    pub fn into_pattern(self) -> (usize, usize) {
        (self.outputs, self.dmodel)
    }

    pub fn neuron(d_model: usize) -> Self {
        Self::new(1, d_model)
    }

    pub fn dmodel(&self) -> usize {
        self.dmodel
    }

    pub fn features(&self) -> usize {
        self.outputs
    }

    pub fn uniform_scale<T: num::Float>(&self) -> T {
        T::from(self.dmodel()).unwrap().recip().sqrt()
    }
}

impl core::fmt::Display for Features {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({}, {})", self.features(), self.dmodel(),)
    }
}

impl IntoDimension for Features {
    type Dim = Ix2;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix2(self.features(), self.dmodel())
    }
}

impl PartialEq<(usize, usize)> for Features {
    fn eq(&self, other: &(usize, usize)) -> bool {
        self.features() == other.0 && self.dmodel() == other.1
    }
}

macro_rules! impl_from {
    ($($s:ty: $t:ty { $into:expr }),* $(,)?) => {
        $(impl_from!(@impl $s: $t { $into });)*
    };
    (@impl $s:ty: $t:ty { $into:expr }) => {
        impl From<$t> for $s {
            fn from(features: $t) -> Self {
                $into(features)
            }
        }
    };
}

impl_from!(
    Features: usize { |f: usize| Features::new(1, f) },
    Features: [usize; 2] {| shape: [usize; 2] | Features::new(shape[0], shape[1])},
    Features: (usize, usize) {| shape: (usize, usize) | Features::new(shape.0, shape.1)},
    Features: nd::Ix1 {| shape: nd::Ix1 | Features::from(&shape)},
    Features: nd::Ix2 {| shape: nd::Ix2 | Features::from(&shape)},
    Features: nd::IxDyn {| shape: nd::IxDyn | Features::from(&shape)},
);

impl_from!(
    nd::Ix2: Features { |f: Features| f.into_dimension() },
    nd::IxDyn: Features { |f: Features| f.into_dimension().into_dyn() },
    [usize; 2]: Features { |f: Features| [f.outputs, f.dmodel] },
    (usize, usize): Features { |f: Features| (f.outputs, f.dmodel) },
);

impl<'a, D> From<&'a D> for Features
where
    D: RemoveAxis,
{
    fn from(dim: &'a D) -> Features {
        features(dim.clone()).unwrap()
    }
}
