/*
    Appellation: params <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![allow(unused_imports)]
extern crate concision_core as cnc;
extern crate concision_linear as linear;

use cnc::Predict;
use core::str::FromStr;
use linear::Features;
use linear::params::{LinearParams, Param, Unbiased};
use ndarray::prelude::*;

const SAMPLES: usize = 20;
const D_MODEL: usize = 5;
const FEATURES: usize = 3;

#[test]
fn test_keys() {
    for i in [(Param::Bias, "bias"), (Param::Weight, "weight")].iter() {
        let kind = Param::from_str(i.1).unwrap();
        assert_eq!(i.0, kind);
    }
}

#[test]
fn test_builders() {
    let shape = (D_MODEL, FEATURES);
    let params = LinearParams::<f64>::ones(shape);
    assert!(params.is_biased());
    assert_eq!(params.weights(), &Array2::ones(shape));
    assert_eq!(params.bias(), &Array1::ones(D_MODEL));
    let params = LinearParams::<usize, Unbiased>::zeros(shape);
    assert!(!params.is_biased());
    assert_eq!(params.weights(), &Array2::zeros(shape));
}

#[test]
#[cfg(feature = "rand")]
fn test_linear_params() {
    let (samples, inputs, outputs) = (SAMPLES, D_MODEL, FEATURES);
    let features = Features::new(outputs, inputs);
    let data = Array2::<f64>::ones((samples, inputs));
    let params = LinearParams::biased(features).uniform();
    let y: Array2<f64> = params.predict(&data).unwrap();
    assert_eq!(y.dim(), (samples, outputs));
}
