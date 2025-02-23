/*
    Appellation: norm <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_linear as linear;

use concision::{Forward, linarr};
use linear::{Biased, LayerNorm};

use approx::assert_abs_diff_eq;
use lazy_static::lazy_static;
use ndarray::prelude::*;

const SHAPE: (usize, usize) = (3, 3);

lazy_static! {
    static ref NORM: Array2<f64> = array![[-0.5492, -0.1619, 0.2254], [0.6127, 1.0000, 1.3873], [
        1.7746, 2.1619, 2.5492
    ],];
}

#[test]
fn test_layer_norm() {
    let shape = SHAPE;
    let x = linarr::<f64, Ix2>(shape).unwrap();

    let ln = LayerNorm::<f64, Biased>::ones(shape);
    let y = ln.forward(&x).expect("LayerNorm forward failed");

    assert_eq!(y.dim(), shape);
    assert_abs_diff_eq!(y, *NORM, epsilon = 1e-4);
}
