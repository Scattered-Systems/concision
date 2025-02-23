/*
    Appellation: primitives <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::params::*;
use nd::{ArrayBase, Ix0, Ix1};

pub(crate) type Pair<A, B = A> = (A, B);

pub(crate) type MaybePair<A, B = A> = Pair<A, Option<B>>;

pub(crate) type NodeBase<S, D = Ix1, E = Ix0> = MaybePair<ArrayBase<S, D>, ArrayBase<S, E>>;

pub(crate) type Node<A = f64, D = Ix1, E = Ix0> = NodeBase<nd::OwnedRepr<A>, D, E>;

pub(crate) mod params {

    macro_rules! params_ty {
        ($($name:ident<$repr:ident>),* $(,)?) => {
            $(params_ty!(@impl $name<$repr>);)*
        };
        (@impl $name:ident<$repr:ident>) => {
            pub type $name<T = f64, K = $crate::params::Biased, D = ndarray::Ix2> = $crate::params::ParamsBase<ndarray::$repr<T>, D, K>;
        };
    }

    params_ty!(LinearParams<OwnedRepr>, LinearParamsShared<OwnedArcRepr>,);
}
