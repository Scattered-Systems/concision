/*
   Appellation: features <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Ix2;
use ndarray::IntoDimension;
use serde::{Deserialize, Serialize};

pub trait Features {
    fn inputs(&self) -> usize;

    fn outputs(&self) -> usize;

    fn in_by_out(&self) -> (usize, usize) {
        (self.inputs(), self.outputs())
    }

    fn out_by_in(&self) -> (usize, usize) {
        (self.outputs(), self.inputs())
    }

    fn input_scale<T: num::Float>(&self) -> T {
        (T::one() / T::from(self.inputs()).unwrap()).sqrt()
    }
}

pub trait FeaturesExt: Features + IntoDimension<Dim = Ix2> {
    fn new(inputs: usize, outputs: usize) -> Self;

    fn single(inputs: usize) -> Self
    where
        Self: Sized,
    {
        Self::new(inputs, 1)
    }
}

// impl<T> FeaturesExt for T
// where
//     T: Features + IntoDimension<Dim = Ix2>,
// {
//     fn new(inputs: usize, outputs: usize) -> Self {
//         Self::from_dimension(ndarray::Ix2(outputs, inputs))
//     }
// }

pub trait FromFeatures<Sh: Features> {
    fn from_features(features: LayerShape) -> Self;
}

pub trait IntoFeatures {
    fn into_features(self) -> LayerShape;
}

impl<S> IntoFeatures for S
where
    S: Into<LayerShape>,
{
    fn into_features(self) -> LayerShape {
        self.into()
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct LayerShape {
    pub inputs: usize,
    pub outputs: usize,
}

impl LayerShape {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self { inputs, outputs }
    }

    pub fn neuron(inputs: usize) -> Self {
        Self::new(inputs, 1)
    }

    pub fn uniform_scale<T: num::Float>(&self) -> T {
        (T::one() / T::from(self.inputs()).unwrap()).sqrt()
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    pub fn in_by_out(&self) -> (usize, usize) {
        (self.inputs, self.outputs)
    }

    pub fn out_by_in(&self) -> (usize, usize) {
        (self.outputs, self.inputs)
    }
}

impl std::fmt::Display for LayerShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.inputs, self.outputs)
    }
}

// impl From<LayerShape> for (usize,) {
//     fn from(features: LayerShape) -> Self {
//         (features.inputs,)
//     }
// }

impl Features for LayerShape {
    fn inputs(&self) -> usize {
        self.inputs
    }

    fn outputs(&self) -> usize {
        self.outputs
    }
}

impl IntoDimension for LayerShape {
    type Dim = ndarray::Ix2;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix2(self.outputs, self.inputs)
    }
}

impl From<LayerShape> for ndarray::Ix2 {
    fn from(features: LayerShape) -> Self {
        ndarray::Ix2(features.outputs, features.inputs)
    }
}

impl From<LayerShape> for ndarray::IxDyn {
    fn from(features: LayerShape) -> Self {
        ndarray::IxDyn(&[features.outputs, features.inputs])
    }
}

impl From<LayerShape> for [usize; 2] {
    fn from(features: LayerShape) -> Self {
        [features.outputs, features.inputs]
    }
}

impl From<[usize; 2]> for LayerShape {
    fn from(features: [usize; 2]) -> Self {
        Self {
            inputs: features[0],
            outputs: features[1],
        }
    }
}

impl From<LayerShape> for (usize, usize) {
    fn from(features: LayerShape) -> Self {
        (features.outputs, features.inputs)
    }
}

impl From<(usize, usize)> for LayerShape {
    fn from((inputs, outputs): (usize, usize)) -> Self {
        Self { inputs, outputs }
    }
}

impl From<usize> for LayerShape {
    fn from(inputs: usize) -> Self {
        Self { inputs, outputs: 1 }
    }
}
