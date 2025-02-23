/*
    Appellation: mlp <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as cnc;
extern crate concision_linear as linear;

use cnc::prelude::{Forward, ReLU, linarr};
use linear::mlp::Perceptron;
use linear::{Biased, Features, Linear};
use ndarray::prelude::*;

#[test]
#[cfg(feature = "rand")]
fn test_perceptron() {
    use cnc::InitializeExt;
    let samples = 100;
    let features = Features::new(1, 300);
    let data = linarr::<f64, Ix2>((samples, features.dmodel())).unwrap();
    let layer = Linear::<f64, Biased>::lecun_normal(features, 1);
    let mlp = Perceptron::new(layer.clone(), Box::new(ReLU::relu));
    assert_eq!(
        mlp.forward(&data).unwrap(),
        layer.forward(&data).unwrap().relu()
    );
}
