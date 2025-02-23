/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::LinearConfig;
use super::layout::{Features, Layout};
use crate::params::{Biased, Unbiased};
use core::marker::PhantomData;
use nd::prelude::*;
use nd::{IntoDimension, RemoveAxis, ShapeError};

impl<K, D> LinearConfig<K, D>
where
    D: Dimension,
{
    pub fn new() -> Self {
        Self {
            layout: Layout::default(),
            name: String::new(),
            _biased: PhantomData::<K>,
        }
    }

    pub fn from_dim(dim: D) -> Result<Self, ShapeError>
    where
        D: Dimension,
    {
        let layout = Layout::from_dim(dim)?;
        let res = Self::new().with_layout(layout);
        Ok(res)
    }

    pub fn from_shape<Sh>(shape: Sh) -> Self
    where
        D: RemoveAxis,
        Sh: ShapeBuilder<Dim = D>,
    {
        let layout = Layout::from_shape(shape);
        Self::new().with_layout(layout)
    }

    pub fn into_biased(self) -> LinearConfig<Biased, D> {
        LinearConfig {
            layout: self.layout,
            name: self.name,
            _biased: PhantomData::<Biased>,
        }
    }

    pub fn into_unbiased(self) -> LinearConfig<Unbiased, D> {
        LinearConfig {
            layout: self.layout,
            name: self.name,
            _biased: PhantomData::<Unbiased>,
        }
    }

    pub fn with_name(self, name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            ..self
        }
    }

    pub fn with_layout<E>(self, layout: Layout<E>) -> LinearConfig<K, E>
    where
        E: Dimension,
    {
        LinearConfig {
            layout,
            name: self.name,
            _biased: self._biased,
        }
    }

    pub fn with_shape<E, Sh>(self, shape: Sh) -> LinearConfig<K, E>
    where
        E: RemoveAxis,
        Sh: ShapeBuilder<Dim = E>,
    {
        LinearConfig {
            layout: self.layout.with_shape(shape),
            name: self.name,
            _biased: self._biased,
        }
    }

    /// This function attempts to convert the [layout](Layout) of the [Config] into a new [dimension](ndarray::Dimension)
    pub fn into_dimensionality<E>(self, dim: E) -> Result<LinearConfig<K, E>, nd::ShapeError>
    where
        E: Dimension,
    {
        let tmp = LinearConfig {
            layout: self.layout.into_dimensionality(dim)?,
            name: self.name,
            _biased: self._biased,
        };
        Ok(tmp)
    }
    /// Determine whether the [Config] is [Biased];
    /// Returns true by comparing the [TypeId](core::any::TypeId) of `K` against the [TypeId](core::any::TypeId) of the [Biased] type
    pub fn is_biased(&self) -> bool
    where
        K: 'static,
    {
        use core::any::TypeId;

        TypeId::of::<K>() == TypeId::of::<Biased>()
    }
    /// Returns an instance to the [Features] of the [Layout]
    pub fn features(&self) -> Features {
        self.layout().features()
    }
    /// Returns an owned reference to the [Layout]
    pub const fn layout(&self) -> &Layout<D> {
        &self.layout
    }
    /// Returns an immutable reference to the `name` of the model.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a cloned reference to the [dimension](ndarray::Dimension) of the [layout](Layout)
    pub fn dim(&self) -> D {
        self.layout().dim()
    }

    pub fn into_pattern(self) -> D::Pattern {
        self.dim().into_pattern()
    }

    pub fn ndim(&self) -> usize {
        self.layout().ndim()
    }
}

impl<K> LinearConfig<K, Ix2> {
    pub fn std(inputs: usize, outputs: usize) -> Self {
        Self {
            layout: Layout::new((outputs, inputs).into_dimension()),
            name: String::new(),
            _biased: PhantomData::<K>,
        }
    }
}

impl<D> LinearConfig<Biased, D>
where
    D: Dimension,
{
    /// The default constructor method for building [Biased] configurations.
    pub fn biased() -> Self {
        Self::new()
    }
}

impl<D> LinearConfig<Unbiased, D>
where
    D: Dimension,
{
    pub fn unbiased() -> Self {
        Self::new()
    }
}

impl<K, D> concision::Config for LinearConfig<K, D> where D: Dimension {}

impl<D> Default for LinearConfig<Biased, D>
where
    D: Dimension,
{
    fn default() -> Self {
        Self::new()
    }
}
