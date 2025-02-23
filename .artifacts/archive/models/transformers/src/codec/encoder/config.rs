/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub struct EncoderConfig {
    pub layers: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self { layers: crate::N }
    }
}
