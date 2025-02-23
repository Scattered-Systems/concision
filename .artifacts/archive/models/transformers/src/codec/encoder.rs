/*
    Appellation: encoder <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{config::EncoderConfig, layer::EncoderLayer};

pub mod config;
pub mod layer;

use linear::norm::LayerNorm;

#[derive(Default)]
pub struct Encoder {
    config: EncoderConfig,
    layers: Vec<EncoderLayer>,
    norm: LayerNorm,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            config: EncoderConfig::default(),
            layers: Vec::new(),
            norm: LayerNorm::default(),
        }
    }

    pub const fn config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn layers(&self) -> &[EncoderLayer] {
        &self.layers
    }

    pub fn norm(&self) -> &LayerNorm {
        &self.norm
    }
}
