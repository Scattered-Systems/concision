/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{item::*, qkv::QkvBase};

mod qkv;

pub mod item;

macro_rules! params_ty {
    ($target:ident {$($name:ident: $(&$lt:lifetime)? $repr:ident),* $(,)?}) => {
        $(params_ty!(@impl $target: $name<$(&$lt)? $repr>);)*
    };
    (@impl $target:ident: $name:ident<$repr:ident>) => {
        pub type $name<A = f64, D = ndarray::Ix2> = $target<ndarray::$repr<A>, D>;
    };
    (@impl $target:ident: $name:ident<&'a $repr:ident>) => {
        pub type $name<'a, A = f64, D = ndarray::Ix2> = $target<ndarray::$repr<&'a A>, D>;
    };
}

params_ty!(
    QkvBase {
        Qkv: OwnedRepr,
        ArcQkv: OwnedArcRepr,
        ViewQkv: &'a ViewRepr,

    }
);

#[allow(unused_imports)]
pub(crate) mod prelude {
    pub use super::item::QKV;
    pub use super::qkv::QkvBase;
    pub use super::{ArcQkv, Qkv, ViewQkv};
}
