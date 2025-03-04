/*
   Appellation: utils <test>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as cnc;

use cnc::linarr;
use ndarray::prelude::*;

#[test]
fn test_conj() {
    use cnc::Conjugate;
    use num::complex::Complex;
    let data = (1..5).map(|x| x as f64).collect::<Vec<_>>();
    let a = Array2::from_shape_vec((2, 2), data).unwrap();
    let exp = array![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(exp, a.conj());

    let a = array![
        [Complex::new(0.0, 0.0), Complex::new(1.0, 0.25)],
        [Complex::new(2.0, 0.5), Complex::new(3.0, 0.0)]
    ];

    let exp = array![
        [Complex::new(0.0, 0.0), Complex::new(1.0, -0.25)],
        [Complex::new(2.0, -0.5), Complex::new(3.0, 0.0)]
    ];

    assert_eq!(exp, a.conj());
}

#[test]
fn test_inverse() {
    use cnc::Inverse;
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[1.0, 2.0, 3.0,], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let exp = array![[-2.0, 1.0], [1.5, -0.5]];
    assert_eq!(Some(exp), a.inverse());
    assert_eq!(None, b.inverse());
}

#[test]
fn test_linarr() {
    let shape = (2, 3);
    let n = shape.0 * shape.1;
    let args = linarr::<f64, Ix2>(shape.clone()).unwrap();
    let exp = Array::linspace(0f64, (n - 1) as f64, n)
        .into_shape(shape)
        .unwrap();
    assert_eq!(args, exp);
}

#[test]
fn test_tril() {
    let a = linarr::<f64, Ix2>((3, 3)).unwrap();
    let exp = array![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [6.0, 7.0, 8.0,]];
    assert_eq!(exp, cnc::tril(&a));
}
