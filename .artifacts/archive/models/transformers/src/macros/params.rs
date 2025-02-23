/*
    Appellation: params <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! qkv_view {
    ($method:ident$(.$call:ident)?::$($rest:tt)*) => {
        qkv_view!(@impl $method$(.$call)?::$($rest)*);
    };
    (@impl $method:ident::$($rest:tt)*) => {
        qkv_view!(@impl $method.$method::$($rest)*);
    };
    (@impl $method:ident.$call:ident::<$view:ident>(self) $($rest:tt)*) => {
        pub fn $method(self) -> $crate::params::QkvBase<$view<A>, D> $($rest)* {
            qkv_view!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(mut self) $($rest:tt)*) => {
        pub fn $method(mut self) -> $crate::params::QkvBase<$view<A>, D> $($rest)* {
            qkv_view!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&self) $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::QkvBase<$view<A>, D> $($rest)* {
            qkv_view!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&mut self) $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::QkvBase<$view<A>, D> $($rest)* {
            qkv_view!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&self) $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::QkvBase<$view<&'_ A>, D> $($rest)* {
            qkv_view!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&mut self) $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::QkvBase<$view<&'_ mut A>, D> $($rest)* {
            qkv_view!(@apply $call(self))
        }
    };
    (@apply $call:ident($self:expr)) => {
        $crate::params::QkvBase {
            q: $self.q.$call(),
            k: $self.k.$call(),
            v: $self.v.$call(),
        }
    };
}
