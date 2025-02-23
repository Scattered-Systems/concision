/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub(crate) fn dk(d_model: usize, heads: usize) -> usize {
    d_model / heads
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct AttentionConfig {
    pub d_model: usize, // embedding size; default is 512
    pub heads: usize,   // number of heads; default is 8
}

impl AttentionConfig {
    pub fn new(d_model: usize, heads: usize) -> Self {
        Self { d_model, heads }
    }
    ///
    pub fn create() -> ConfigBuilder {
        ConfigBuilder::new()
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn dk(&self) -> usize {
        dk(self.d_model(), self.heads())
    }

    pub fn heads(&self) -> usize {
        self.heads
    }
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            d_model: crate::D_MODEL,
            heads: crate::HEADS,
        }
    }
}

concision::builder! {
    ConfigBuilder(AttentionConfig) {
        d_model: usize,
        heads: usize,
    }
}
