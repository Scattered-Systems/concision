/*
   Appellation: traits <test>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as cnc;

use cnc::linarr;
use ndarray::{Array2, Ix2, array};

#[test]
fn test_affine() {
    use cnc::Affine;
    let x = array![[0.0, 1.0], [2.0, 3.0]];

    let y = x.affine(4.0, -2.0);
    assert_eq!(y, array![[-2.0, 2.0], [6.0, 10.0]]);
}

#[test]
fn test_masked_fill() {
    use cnc::MaskFill;
    let shape = (2, 2);
    let mask = array![[true, false], [false, true]];
    let arr = linarr::<f64, Ix2>(shape).unwrap();
    let a = arr.masked_fill(&mask, 0.0);
    assert_eq!(a, array![[0.0, 1.0], [2.0, 0.0]]);
}

#[test]
fn test_matrix_power() {
    use cnc::Matpow;
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(x.pow(0), Array2::<f64>::eye(2));
    assert_eq!(x.pow(1), x);
    assert_eq!(x.pow(2), x.dot(&x));
}

#[test]
fn test_unsqueeze() {
    use cnc::Unsqueeze;
    let arr = array![1, 2, 3, 4];
    let a = arr.clone().unsqueeze(0);
    assert_eq!(a.dim(), (1, 4));
    let b = arr.unsqueeze(1);
    assert_eq!(b.dim(), (4, 1));
}
