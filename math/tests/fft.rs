/*
    Appellation: fft <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_math as concision;

use approx::assert_abs_diff_eq;
use concision::signal::fourier::*;
use lazy_static::lazy_static;
use num::complex::{Complex, ComplexFloat};
use num::traits::Float;

const EPSILON: f64 = 1e-6;

lazy_static! {
    static ref EXPECTED_RFFT: Vec<Complex<f64>> = vec![
        Complex { re: 28.0, im: 0.0 },
        Complex { re: -4.0, im: 0.0 },
        Complex {
            re: -4.0,
            im: 1.6568542494923806
        },
        Complex {
            re: -4.0,
            im: 4.000000000000001
        },
        Complex {
            re: -3.999999999999999,
            im: 9.656854249492381
        }
    ];
}

fn handle<S, T>(data: Vec<S>) -> Vec<S>
where
    S: ComplexFloat<Real = T>,
    T: Copy + Float,
{
    let tmp = {
        let mut inner = data
            .iter()
            .cloned()
            .filter(|i| i.im() > T::zero())
            .map(|i| i.conj())
            .collect::<Vec<_>>();
        inner.sort_by(|a, b| a.im().partial_cmp(&b.im()).unwrap());
        inner
    };
    let mut out = data.clone();
    out.sort_by(|a, b| a.re().partial_cmp(&b.re()).unwrap());
    out.sort_by(|a, b| a.im().partial_cmp(&b.im()).unwrap());
    out.extend(tmp);
    out
}

#[test]
#[ignore = "Needs to be fixed"]
fn test_rfft() {
    let polynomial = (0..8).map(|i| i as f64).collect::<Vec<_>>();
    let plan = FftPlan::new(polynomial.len()).build();
    println!("Function Values: {:?}\nPlan: {:?}", &polynomial, &plan);
    let fft = rfft(&polynomial, &plan);
    let res = handle(fft.clone());
    assert!(fft.len() == EXPECTED_RFFT.len());
    for (x, y) in fft.iter().zip(EXPECTED_RFFT.iter()) {
        assert_abs_diff_eq!(x.re(), y.re());
        assert_abs_diff_eq!(x.im(), y.im());
    }
    let plan = FftPlan::new(fft.len()).build();
    let _ifft = dbg!(irfft(&res, &plan));
    // for (x, y) in ifft.iter().zip(polynomial.iter()) {
    //     assert_abs_diff_eq!(*x, *y, epsilon = EPSILON);
    // }
}

#[test]
fn small_polynomial_returns_self() {
    let polynomial = vec![1.0f64, 1.0, 0.0, 2.5];
    let permutation = FftPlan::new(polynomial.len()).build();
    let fft = fft(&polynomial, &permutation);
    let ifft = ifft(&fft, &permutation)
        .into_iter()
        .map(|i| i.re())
        .collect::<Vec<_>>();
    for (x, y) in ifft.iter().zip(polynomial.iter()) {
        assert_abs_diff_eq!(*x, *y, epsilon = EPSILON);
    }
}

#[test]
fn square_small_polynomial() {
    let mut polynomial = vec![1.0f64, 1.0, 0.0, 2.0];
    polynomial.append(&mut vec![0.0; 4]);
    let plan = FftPlan::new(polynomial.len()).build();
    let mut fft = fft(&polynomial, &plan);
    fft.iter_mut().for_each(|num| *num *= *num);
    let ifft = ifft(&fft, &plan)
        .into_iter()
        .map(|i| i.re())
        .collect::<Vec<_>>();
    let expected = [1.0, 2.0, 1.0, 4.0, 4.0, 0.0, 4.0, 0.0, 0.0];
    for (x, y) in ifft.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*x, *y, epsilon = EPSILON);
    }
}

#[test]
#[ignore]
fn square_big_polynomial() {
    // This test case takes ~1050ms on my machine in unoptimized mode,
    // but it takes ~70ms in release mode.
    let n = 1 << 17; // ~100_000
    let mut polynomial = vec![1.0f64; n];
    polynomial.append(&mut vec![0.0f64; n]);
    let permutation = FftPlan::new(polynomial.len()).build();
    let mut fft = fft(&polynomial, &permutation);
    fft.iter_mut().for_each(|num| *num *= *num);
    let ifft = irfft(&fft, &permutation)
        .into_iter()
        .map(|i| i.re())
        .collect::<Vec<_>>();
    let expected = (0..((n << 1) - 1)).map(|i| std::cmp::min(i + 1, (n << 1) - 1 - i) as f64);
    for (&x, y) in ifft.iter().zip(expected) {
        assert_abs_diff_eq!(x, y);
    }
}
