/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::EPSILON;
use nd::prelude::{Axis, Dimension, Ix2};

pub struct Config<D = Ix2> {
    pub axis: Option<Axis>,
    pub dim: D,
    pub eps: f64,
}

impl<D> Config<D>
where
    D: Dimension,
{
    pub fn new() -> ConfigBuilder<D> {
        ConfigBuilder::new()
    }

    pub fn axis(&self) -> Option<&Axis> {
        self.axis.as_ref()
    }

    pub fn axis_mut(&mut self) -> &mut Option<Axis> {
        &mut self.axis
    }

    pub const fn eps(&self) -> f64 {
        self.eps
    }

    pub fn eps_mut(&mut self) -> &mut f64 {
        &mut self.eps
    }

    pub fn dim(&self) -> D::Pattern {
        self.raw_dim().into_pattern()
    }

    pub fn dim_mut(&mut self) -> &mut D {
        &mut self.dim
    }

    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    pub fn raw_dim(&self) -> D {
        self.dim.clone()
    }

    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }

    pub fn shape_mut(&mut self) -> &mut [usize] {
        self.dim.slice_mut()
    }
}

impl<D> Default for Config<D>
where
    D: Default,
{
    fn default() -> Self {
        Self {
            axis: None,
            dim: D::default(),
            eps: EPSILON,
        }
    }
}

pub struct ConfigBuilder<D = Ix2> {
    axis: Option<Axis>,
    dim: D,
    eps: f64,
}

impl<D> ConfigBuilder<D>
where
    D: Dimension,
{
    pub fn new() -> Self {
        Self {
            axis: None,
            dim: D::default(),
            eps: 1e-5,
        }
    }

    pub fn axis(mut self, axis: Axis) -> Self {
        self.axis = Some(axis);
        self
    }

    pub fn dim(mut self, dim: D) -> Self {
        self.dim = dim;
        self
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn build(self) -> Config<D> {
        Config {
            axis: self.axis,
            dim: self.dim,
            eps: self.eps,
        }
    }
}
