/*
   Appellation: macros <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#[macro_use]
mod activate;
#[macro_use]
mod getters;
#[macro_use]
mod ops;

/// Generates methods for forwarding [dimensional](ndarray::Dimension) related methods inherited from
/// a particular field (or method via `$name()`) which references either an [ArrayBase](ndarray::ArrayBase)
/// or another `type` implementing
#[macro_export]
macro_rules! dimensional {
    ($name:ident$($rest:tt)*) => {
        $crate::dimensional!(@impl $name$($rest)*);
    };
    (@impl $name:ident) => {
        /// Return the [pattern](ndarray::Dimension::Pattern) of the dimension
        pub fn dim(&self) -> D::Pattern {
            self.$name.dim()
        }
        /// Returns rank (ndim) of the dimension
        pub fn ndim(&self) -> usize {
            self.$name.ndim()
        }
        /// Forwards the
        pub fn raw_dim(&self) -> D {
            self.$name.raw_dim()
        }
        /// Returns a reference the shape of the dimension as a slice
        pub fn shape(&self) -> &[usize] {
            self.$name.shape()
        }
    };
    (@impl $name:ident()) => {
        /// Return the [pattern](ndarray::Dimension::Pattern) of the dimension
        pub fn dim(&self) -> D::Pattern {
            self.$name().dim()
        }
        /// Returns rank (ndim) of the dimension
        pub fn ndim(&self) -> usize {
            self.$name().ndim()
        }
        /// Forwards the
        pub fn raw_dim(&self) -> D {
            self.$name().raw_dim()
        }
        /// Returns a reference the shape of the dimension as a slice
        pub fn shape(&self) -> &[usize] {
            self.$name().shape()
        }
    };
}

/// Implement methods native to the `ndarray` [dimension](ndarray::Dimension)
#[macro_export]
macro_rules! fwd_ndim {
    ($name:ident$(())?) => {
        fwd_ndim!(@impl $name$(())?);
    };
    (@impl $name:ident) => {
        /// Returns a reference to the current dimension, as a slice.
        pub fn as_slice(&self) -> &[usize] {
            self.$name$(())?.shape()
        }

        pub fn into_pattern(self) -> D::Pattern {
            self.$name$(())?.into_pattern()
        }

        pub fn ndim(&self) -> usize {
            self.$name$(())?.ndim()
        }

        pub fn raw_dim(&self) -> D {
            self.$name$(())?.dim().clone()
        }
    }
}
