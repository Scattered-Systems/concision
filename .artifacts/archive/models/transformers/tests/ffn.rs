/*
    Appellation: ffn <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as cnc;
extern crate concision_linear as linear;
extern crate concision_transformer as transformer;

use cnc::prelude::{Predict, linarr};
use linear::Biased;
use transformer::model::ffn::FeedForwardNetwork;

use ndarray::prelude::*;

#[test]
fn test_ffn() {
    let (samples, d_model, d_ff) = (100, 30, 3);
    #[cfg(feature = "rand")]
    let model = FeedForwardNetwork::<f64, Biased>::std(d_model, d_ff, Some(0.45));
    #[cfg(not(feature = "rand"))]
    let model = FeedForwardNetwork::<f64, Biased>::std(d_model, d_ff);

    let data = linarr::<f64, Ix2>((samples, d_model)).unwrap();

    let pred = model.predict(&data).unwrap();
    assert_eq!(pred.dim(), (samples, d_model));
}
