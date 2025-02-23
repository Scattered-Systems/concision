/*
    Appellation: macros <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[macro_use]
mod params;

macro_rules! ndbuilder {
    ($method:ident$(::$call:ident)?() $($where:tt)*) => {
        ndbuilder!(@impl $method$(::$call)?() $($where)*);
    };
    (@impl $method:ident() $($where:tt)*) => {
        ndbuilder!(@impl $method::$method() $($where)*);
    };
    (@impl $method:ident::$call:ident() $($where:tt)*) => {
        pub fn $method<Sh: ndarray::ShapeBuilder<Dim = D>>(shape: Sh) -> Self $($where)* {
            Self::builder(shape, ndarray::ArrayBase::$call)
        }
    };
}

#[allow(unused_macros)]
macro_rules! cbuilder {
    (@impl derive: [$($D:ident),* $(,)?], $name:ident {$($vis:vis $field:ident: $type:ty),*}) => {
        #[derive(Clone, Debug, PartialEq, $($D),*)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        pub struct $name {
            $($vis $field: $type),*
        }
        impl $name {
            paste::paste! {
                pub fn new() -> [<$name Builder>] {
                    [<$name Builder>]::new()
                }
            }

            $(
                pub fn $field(mut self, $field: $type) -> Self {
                    self.$field = $field;
                    self
                }
            )*
        }
    };
    (@builder derive: [$($D:ident),* $(,)?], $name:ident {$($field:ident: $type:ty),*}) => {
        pub struct $name {
            $(pub(crate) $field: $type),*
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    $($field: None),*
                }
            }

            $(
                pub fn $field(mut self, $field: $type) -> Self {
                    self.$field = Some($field);
                    self
                }
            )*

            pub fn build(&self) -> Config {
                Config {
                    $($field: self.$field.unwrap_or_else(|| crate::$field),)*
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// This macro helps create a stack of identical sublayers.
///
#[allow(unused_macros)]
macro_rules! sublayer {
    (@impl heads: $heads:expr) => {};
}
