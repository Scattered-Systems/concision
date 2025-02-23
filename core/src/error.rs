/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
/// A type alias for a [Result] with the error type [Error].
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, scsys::VariantConstructors, thiserror::Error)]
pub enum Error {
    #[error("Model Error: {0}")]
    ModelError(#[from] ModelError),
    #[error("Shape Error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),
    #[error("Unknown Error: {0}")]
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, scsys::VariantConstructors, thiserror::Error)]
pub enum ModelError {
    #[error("Prediction Error: {0}")]
    PredictError(String),
}
