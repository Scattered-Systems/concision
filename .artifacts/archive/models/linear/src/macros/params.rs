/*
    Appellation: params <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! pbuilder {
    ($method:ident$(.$call:ident)? where $($rest:tt)*) => {
        pbuilder!(@impl $method$(.$call)? where $($rest)*);
    };
    (@impl $method:ident where $($rest:tt)*) => {
        pbuilder!(@impl $method.$method where $($rest)*);
    };
    (@impl $method:ident.$call:ident where $($rest:tt)*) => {
        pub fn $method<Sh>(shape: Sh) -> Self
        where
            K: $crate::params::mode::ParamMode,
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            let dim = shape.into_shape_with_order().raw_dim().clone();
            ParamsBase {
                bias: build_bias(K::BIASED, dim.clone(), |dim| ndarray::ArrayBase::$call(dim)),
                weight: ndarray::ArrayBase::$call(dim),
                _mode: ::core::marker::PhantomData::<K>,
            }
        }
    };
}

macro_rules! wnbview {
    ($method:ident::$($rest:tt)*) => {
        wnbview!(@impl $method.$method::$($rest)*);
    };
    ($method:ident.$call:ident::$($rest:tt)*) => {
        wnbview!(@impl $method.$call::$($rest)*);
    };
    (@impl $method:ident.$call:ident::<$view:ident>(self) where $($rest:tt)*) => {
        pub fn $method(self) -> $crate::params::ParamsBase<$view<A>, D, K>
        where
            $($rest)*
        {
            wnbview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(mut self) where $($rest:tt)*) => {
        pub fn $method(mut self) -> $crate::params::ParamsBase<$view<A>, D, K>
        where
            $($rest)*
        {
            wnbview!(@apply $call(self).as_mut())
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::ParamsBase<$view<A>, D, K>
        where
            $($rest)*
        {
            wnbview!(@apply $call(self).as_ref())
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::ParamsBase<$view<A>, D, K>
        where
            $($rest)*
        {
            wnbview!(@apply $call(self).as_mut())
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::ParamsBase<$view<&'_ A>, D, K>
        where
            $($rest)*
        {
            wnbview!(@apply $call(&self).as_ref())
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::ParamsBase<$view<&'_ mut A>, D, K>
        where
            $($rest)*
        {
            wnbview!(@apply $call(self).as_mut())
        }
    };
    (@apply $call:ident($self:expr)$(.$as:ident())?) => {
        $crate::params::ParamsBase {
            bias: $self.bias$(.$as())?.map(|arr| arr.$call()),
            weight: $self.weight.$call(),
            _mode: $self._mode,
        }
    };
}
