/*
    Appellation: codec <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Decoder, Encoder};
use concision::{builder, getters};

#[derive(Default)]
pub struct Codec {
    ctx: Context,
    decoder: Decoder,
    encoder: Encoder,
}

impl Codec {
    pub fn new() -> CodecBuilder {
        CodecBuilder::new()
    }

    getters!(
        context.ctx<Context>,
        decoder<Decoder>,
        encoder<Encoder>,
    );
}

builder! {
    CodecBuilder(Codec) {
        ctx: Context,
        decoder: Decoder,
        encoder: Encoder,
    }
}

#[derive(Default)]
pub struct Generator {
    pub dmodel: usize,
    pub vocab: Vec<String>,
}

#[derive(Default)]
pub struct Context {
    pub src: String, // source embedding
    pub tgt: String, // target embedding
}

impl Context {
    pub fn new() -> Self {
        Self {
            src: String::new(),
            tgt: String::new(),
        }
    }

    pub fn with_src(self, src: String) -> Self {
        Self { src, ..self }
    }

    pub fn with_tgt(self, tgt: String) -> Self {
        Self { tgt, ..self }
    }
}
