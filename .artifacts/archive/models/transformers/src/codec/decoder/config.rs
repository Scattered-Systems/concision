/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub struct DecoderConfig {
    pub layers: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self { layers: crate::N }
    }
}
