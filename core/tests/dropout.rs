extern crate concision_core as cnc;

use cnc::Forward;
use cnc::nn::Dropout;
use ndarray::prelude::*;

#[test]
fn test_dropout() {
    let shape = (512, 2048);
    let arr = Array2::<f64>::ones(shape);
    let dropout = Dropout::new(0.5);
    let out = dropout.forward(&arr);

    assert!(arr.iter().all(|&x| x == 1.0));
    assert!(out.iter().any(|x| x == &0f64));
}
