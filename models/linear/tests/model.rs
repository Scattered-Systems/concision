/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::func::Sigmoid;
use concision::{linarr, Predict};
use linear::{Config, Features, Linear};

use lazy_static::lazy_static;
use ndarray::*;

const SAMPLES: usize = 20;
const INPUTS: usize = 5;
const OUTPUT: usize = 3;

lazy_static! {
    static ref FEATURES: Features = Features::new(OUTPUT, INPUTS);
    static ref CONFIG: Config = Config::new("test", FEATURES.clone());
    static ref SAMPLE_DATA: Array<f64, Ix2> = linarr::<f64, Ix2>((SAMPLES, INPUTS)).unwrap();
    static ref SHAPE: (usize, usize, usize) = (SAMPLES, INPUTS, OUTPUT);
}

#[test]
fn test_linear() {
    let (samples, inputs, outputs) = (20, 5, 3);

    let features = Features::new(outputs, inputs);
    let config = Config::new("test", features.clone()).biased();

    let data = linarr::<f64, Ix2>((samples, inputs)).unwrap();

    let model: Linear<f64> = Linear::std(config).uniform();

    let y = model.activate(&data, Sigmoid::sigmoid).unwrap();
    assert_eq!(y.shape(), &[samples, outputs]);
}
