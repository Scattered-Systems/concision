/*
    Appellation: kinds <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{ArrayBase, Dimension, Ix2, OwnedRepr, RawData};
use strum::{AsRefStr, EnumCount, EnumDiscriminants, EnumIs, VariantNames};

#[derive(AsRefStr, EnumCount, EnumDiscriminants, EnumIs, VariantNames)]
#[strum_discriminants(
    derive(
        AsRefStr,
        EnumCount,
        EnumIs,
        Hash,
        Ord,
        PartialOrd,
        VariantNames,
        strum::Display,
        strum::EnumString,
    ),
    name(QKV),
    strum(serialize_all = "lowercase")
)]
#[cfg_attr(
    feature = "serde",
    strum_discriminants(
        derive(serde::Deserialize, serde::Serialize),
        serde(rename_all = "lowercase", untagged)
    )
)]
#[strum(serialize_all = "lowercase")]
pub enum Entry<S = OwnedRepr<f64>, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    Q(ArrayBase<S, D>),
    K(ArrayBase<S, D>),
    V(ArrayBase<S, D>),
}

impl<A, S, D> Entry<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn from_q(q: ArrayBase<S, D>) -> Self {
        Self::Q(q)
    }

    pub fn from_k(k: ArrayBase<S, D>) -> Self {
        Self::K(k)
    }

    pub fn from_v(v: ArrayBase<S, D>) -> Self {
        Self::V(v)
    }

    pub fn q(&self) -> Option<&ArrayBase<S, D>> {
        match self {
            Self::Q(q) => Some(q),
            _ => None,
        }
    }

    pub fn k(&self) -> Option<&ArrayBase<S, D>> {
        match self {
            Self::K(k) => Some(k),
            _ => None,
        }
    }

    pub fn v(&self) -> Option<&ArrayBase<S, D>> {
        match self {
            Self::V(v) => Some(v),
            _ => None,
        }
    }
}
