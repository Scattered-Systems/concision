/*
   Appellation: traits <test>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_math as cnc;

use ndarray::prelude::*;
use num::Complex;

#[test]
fn test_as_complex() {
    use cnc::AsComplex;
    let x = 1.0;
    let y = x.as_re();
    assert_eq!(y, Complex::new(1.0, 0.0));
}

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
