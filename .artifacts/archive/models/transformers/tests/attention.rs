/*
    Appellation: attention <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_transformer as transformer;

use transformer::AttentionHead;

use ndarray::prelude::*;

#[test]
fn attention_head() {
    let shape = (3, 3);

    let head = AttentionHead::<f64>::ones(shape);
    assert_eq!(head.q(), &Array::ones(shape));
    let exp = Array2::from_elem(shape, 1f64 / 3f64);
    let score = head.attention();
    assert!(score.attention().abs_diff_eq(&exp, 1e-6));
}
