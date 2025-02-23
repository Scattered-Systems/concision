/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::prelude::{Sigmoid, linarr};
use lazy_static::lazy_static;
use linear::params::{Biased, Unbiased};
use linear::{Features, Linear, LinearConfig};
use ndarray::*;

const SAMPLES: usize = 20;
const D_MODEL: usize = 5;
const OUTPUTS: usize = 3;
const SHAPE: (usize, (usize, usize)) = (SAMPLES, (OUTPUTS, D_MODEL));

lazy_static! {
    static ref FEATURES: Features = Features::new(OUTPUTS, D_MODEL);
}

#[test]
fn test_config() {
    let dim = FEATURES.clone().into_dimension();
    let config = LinearConfig::<Biased>::from_shape(dim);
    assert!(config.is_biased());
    let config = LinearConfig::<Unbiased>::from_shape(dim);
    assert!(!config.is_biased());
}

#[test]
fn test_model_toggle() {
    let (_samples, (outputs, inputs)) = SHAPE;

    let model = Linear::<f64>::from_features(inputs, outputs);
    assert!(model.is_biased());

    let model = Linear::<f64, Unbiased>::from_features(inputs, outputs);
    assert!(!model.is_biased());

    let model = Linear::<f64>::from_features(inputs, outputs).into_unbiased();
    assert!(!model.is_biased());
}

#[test]
#[cfg(feature = "rand")]
fn test_linear() {
    let (samples, (outputs, inputs)) = SHAPE;

    let model = Linear::<f64, Biased>::from_features(inputs, outputs).uniform();

    let data = linarr::<f64, Ix2>((samples, inputs)).unwrap();
    let y = model.activate(&data, Sigmoid::sigmoid).unwrap();

    assert_eq!(y.shape(), &[samples, outputs]);
}
