/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::Zero;

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    scsys::VariantConstructors,
    strum::AsRefStr,
    strum::Display,
    strum::EnumCount,
    strum::EnumIs,
    strum::EnumIter,
    strum::EnumString,
    strum::VariantArray,
    strum::VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case", untagged)
)]
#[strum(serialize_all = "snake_case")]
pub enum PadAction {
    Clipping,
    Lane,
    Reflecting,
    #[default]
    StopAfterCopy,
    Wrapping,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    strum::AsRefStr,
    strum::Display,
    strum::EnumCount,
    strum::EnumIs,
    strum::VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
#[strum(serialize_all = "snake_case")]
#[repr(C)]
pub enum PadMode<T = f64> {
    Constant(T),
    Edge,
    Maximum,
    Mean,
    Median,
    Minimum,
    Mode,
    Reflect,
    Symmetric,
    Wrap,
}

impl<T> Default for PadMode<T> {
    fn default() -> Self {
        PadMode::Wrap
    }
}

impl<T> From<T> for PadMode<T> {
    fn from(value: T) -> Self {
        PadMode::Constant(value)
    }
}

impl<T> PadMode<T> {
    pub fn as_pad_action(&self) -> PadAction {
        match *self {
            PadMode::Constant(_) => PadAction::StopAfterCopy,
            PadMode::Edge => PadAction::Clipping,
            PadMode::Maximum => PadAction::Clipping,
            PadMode::Mean => PadAction::Clipping,
            PadMode::Median => PadAction::Clipping,
            PadMode::Minimum => PadAction::Clipping,
            PadMode::Mode => PadAction::Clipping,
            PadMode::Reflect => PadAction::Reflecting,
            PadMode::Symmetric => PadAction::Reflecting,
            PadMode::Wrap => PadAction::Wrapping,
        }
    }

    pub fn into_pad_action(self) -> PadAction {
        match self {
            PadMode::Constant(_) => PadAction::StopAfterCopy,
            PadMode::Edge => PadAction::Clipping,
            PadMode::Maximum => PadAction::Clipping,
            PadMode::Mean => PadAction::Clipping,
            PadMode::Median => PadAction::Clipping,
            PadMode::Minimum => PadAction::Clipping,
            PadMode::Mode => PadAction::Clipping,
            PadMode::Reflect => PadAction::Reflecting,
            PadMode::Symmetric => PadAction::Reflecting,
            PadMode::Wrap => PadAction::Wrapping,
        }
    }
    pub fn init(&self) -> T
    where
        T: Copy + Zero,
    {
        match *self {
            PadMode::Constant(v) => v,
            _ => T::zero(),
        }
    }
}
