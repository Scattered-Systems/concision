/*
    Appellation: model <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! mbuilder {
    ($method:ident$(.$call:ident)? where $($rest:tt)*) => {
        mbuilder!(@impl $method$(.$call)? where $($rest)*);
    };
    (@impl $method:ident where $($rest:tt)*) => {
        mbuilder!(@impl $method.$method where $($rest)*);
    };
    (@impl $method:ident.$call:ident where $($rest:tt)*) => {
        pub fn $method<Sh>(shape: Sh) -> Self
        where
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            Linear::from_params($crate::params::ParamsBase::$call(shape))
        }
    };
}
