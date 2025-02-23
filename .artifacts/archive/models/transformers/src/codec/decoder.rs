/*
    Appellation: decoder <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{config::DecoderConfig, layer::DecoderLayer};

pub mod config;
pub mod layer;

#[derive(Default)]
pub struct Decoder {
    config: DecoderConfig,
    layers: Vec<DecoderLayer>,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            config: DecoderConfig::default(),
            layers: Vec::new(),
        }
    }

    pub const fn config(&self) -> &DecoderConfig {
        &self.config
    }

    pub fn layers(&self) -> &[DecoderLayer] {
        &self.layers
    }
}
