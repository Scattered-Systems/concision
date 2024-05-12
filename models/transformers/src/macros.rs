/*
    Appellation: macros <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! access {
    ($($var:ident),* $(,)?) => {
        $(access!(@impl $var);)*
    };
    ($via:ident::<$($var:ident),* $(,)?>) => {
        $(access!(@impl $via::$var);)*
    };
    (@impl $var:ident) => {
        pub fn $var(&self) -> &ArrayBase<S, D> {
            &self.$var
        }
        paste::paste! {
            pub fn [< $var _mut>](&mut self) -> &mut ArrayBase<S, D> {
                &mut self.$var
            }
        }
    };
    (@impl $via:ident::$var:ident) => {
        pub fn $var(&self) -> &ArrayBase<S, D> {
            &self.$via.$var
        }
        paste::paste! {
            pub fn [< $var _mut>](&mut self) -> &mut ArrayBase<S, D> {
                &mut self.$via.$var
            }
        }
    };
}

macro_rules! ndbuilder {
    ($method:ident$(::$call:ident)?() where $($rest:tt)*) => {
        ndbuilder!(@impl $method$(::$call)?() where $($rest)*);
    };
    (@impl $method:ident() where $($rest:tt)*) => {
        ndbuilder!(@impl $method::$method() where $($rest)*);
    };
    (@impl $method:ident::$call:ident() where $($rest:tt)*) => {
        pub fn $method<Sh: ndarray::ShapeBuilder<Dim = D>>(shape: Sh) -> Self where $($rest)* {
            Self::builder(shape, ArrayBase::$call)
        }
    };
}

// # TODO:
macro_rules! ndview {
    ($method:ident::$($rest:tt)*) => {
        ndview!(@impl $method.$method::$($rest)*);
    };
    ($method:ident.$call:ident::$($rest:tt)*) => {
        ndview!(@impl $method.$call::$($rest)*);
    };
    (@impl $method:ident.$call:ident::<$view:ident>(self) where $($rest:tt)*) => {
        pub fn $method(self) -> $crate::params::ParamsBase<$view<A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(mut self) where $($rest:tt)*) => {
        pub fn $method(mut self) -> $crate::params::ParamsBase<$view<A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::ParamsBase<$view<A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::ParamsBase<$view<A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::ParamsBase<$view<&'_ A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::ParamsBase<$view<&'_ mut A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@apply $call:ident($self:expr)) => {
        $crate::params::ParamsBase {
            q: $self.q.$call(),
            k: $self.k.$call(),
            v: $self.v.$call(),
        }
    };
}
