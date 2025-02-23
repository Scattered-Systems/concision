/*
    Appellation: error <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub type PadResult<T = ()> = Result<T, PadError>;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, thiserror::Error)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(feature = "std", derive(std::hash::Hash))]
#[repr(u8)]
pub enum PadError {
    #[error("Inconsistent Dimensions: {0}")]
    InconsistentDimensions(String),
}
