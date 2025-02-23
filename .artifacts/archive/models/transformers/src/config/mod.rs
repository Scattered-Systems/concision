/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::getters;

pub struct TransformerConfig {
    pub dropout: Option<f64>,
    pub features: Features,
    pub heads: usize,
    pub layers: usize,
}

impl TransformerConfig {
    pub fn new(dropout: Option<f64>, features: Features, heads: usize, layers: usize) -> Self {
        Self {
            dropout,
            features,
            heads,
            layers,
        }
    }

    getters!(
        dropout<Option<f64>>,
        features<Features>,
        heads<usize>,
        layers<usize>
    );
    getters!(features::<[d_model<usize>, qkv<QkvShape>]>);
    getters!(features::<[dk, dq, dv]> => usize);
}

pub struct Features {
    pub d_model: usize,
    pub qkv: QkvShape,
}

impl Features {
    pub fn new(d_model: usize, qkv: QkvShape) -> Self {
        Self { d_model, qkv }
    }

    getters!(d_model<usize>, qkv<QkvShape>);
    getters!(qkv::<[dk, dq, dv]> => usize);
}

pub struct QkvShape {
    pub dq: usize,
    pub dk: usize,
    pub dv: usize,
}

impl QkvShape {
    pub fn new(dq: usize, dk: usize, dv: usize) -> Self {
        Self { dq, dk, dv }
    }

    pub fn std(dk: usize) -> Self {
        let (dq, dv) = (dk, dk);

        Self::new(dq, dk, dv)
    }

    getters!(dk, dq, dv => usize);
}

impl From<usize> for QkvShape {
    fn from(dk: usize) -> Self {
        Self::std(dk)
    }
}

impl From<(usize, usize, usize)> for QkvShape {
    fn from((dq, dk, dv): (usize, usize, usize)) -> Self {
        Self::new(dq, dk, dv)
    }
}
