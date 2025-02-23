/*
   Appellation: utils <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::FftPlan;
use crate::AsComplex;
use num::complex::{Complex, ComplexFloat};
use num::traits::{Float, FloatConst, NumAssignOps, NumCast, NumOps};

pub(crate) fn fft_angle<T>(n: usize) -> T
where
    T: FloatConst + NumCast + NumOps,
{
    T::TAU() / T::from(n).unwrap()
}

/// Computes the Fast Fourier Transform of a one-dimensional, complex-valued signal.
pub fn fft<S, T>(input: impl AsRef<[S]>, permute: &FftPlan) -> Vec<Complex<S::Real>>
where
    S: ComplexFloat<Real = T>,
    S::Real: Float + FloatConst,
    Complex<S::Real>: ComplexFloat<Real = S::Real> + NumOps<S> + NumOps<S::Real>,
{
    //
    let input = input.as_ref();
    //
    let n = input.len();
    // initialize the result vector
    let mut result = Vec::with_capacity(n);
    // store the input values in the result vector according to the permutation
    for position in permute.clone().into_iter() {
        let arg = input[position];
        result.push(Complex::new(arg.re(), arg.im()));
    }
    let mut segment: usize = 1;
    while segment < n {
        segment <<= 1;
        // compute the angle of the complex number
        let angle = fft_angle::<T>(segment);
        // compute the radius of the complex number (length)
        let radius = Complex::new(angle.cos(), angle.sin());
        // iterate over the signal in segments of length `segment`
        for start in (0..n).step_by(segment) {
            let mut w = Complex::new(T::one(), T::zero());
            for position in start..(start + segment / 2) {
                let a = result[position];
                let b = result[position + segment / 2] * w;
                result[position] = a + b;
                result[position + segment / 2] = a - b;
                w = w * radius;
            }
        }
    }
    result
}

/// Computes the Fast Fourier Transform of an one-dimensional, real-valued signal.
/// TODO: Optimize the function to avoid unnecessary computation.
pub fn rfft<T>(input: impl AsRef<[T]>, input_permutation: impl AsRef<[usize]>) -> Vec<Complex<T>>
where
    T: Float + FloatConst,
    Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
{
    // create a reference to the input
    let input = input.as_ref();
    // fetch the length of the input
    let n = input.len();
    // compute the size of the result vector
    let size = (n - (n % 2)) / 2 + 1;
    // initialize the output vector
    let mut store = Vec::with_capacity(size);
    // store the input values in the result vector according to the permutation
    for position in input_permutation.as_ref() {
        store.push(input[*position].as_re());
    }
    let mut segment: usize = 1;
    while segment < n {
        segment <<= 1;
        // compute the angle of the complex number
        let angle = fft_angle::<T>(segment);
        // compute the radius of the complex number (length)
        let radius = Complex::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(segment) {
            let mut w = Complex::new(T::one(), T::zero());
            for position in start..(start + segment / 2) {
                let a = store[position];
                let b = store[position + segment / 2] * w;
                store[position] = a + b;
                store[position + segment / 2] = a - b;
                w *= radius;
            }
        }
    }
    store
        .iter()
        .cloned()
        .filter(|x| x.im() >= T::zero())
        .collect()
}
/// Computes the Inverse Fast Fourier Transform of an one-dimensional, complex-valued signal.
pub fn ifft<S, T>(input: &[S], input_permutation: &FftPlan) -> Vec<Complex<T>>
where
    S: ComplexFloat<Real = T>,
    T: Float + FloatConst,
    Complex<T>: ComplexFloat<Real = T> + NumOps<S> + NumOps<T>,
{
    let n = input.len();
    let mut result = Vec::with_capacity(n);
    for position in input_permutation.clone().into_iter() {
        let arg = input[position];
        result.push(Complex::new(arg.re(), arg.im()));
    }
    let mut length: usize = 1;
    while length < n {
        length <<= 1;
        let angle = fft_angle::<T>(length).neg();
        let radius = Complex::new(T::cos(angle), T::sin(angle)); // w_len
        for start in (0..n).step_by(length) {
            let mut w = Complex::new(T::one(), T::zero());
            for position in start..(start + length / 2) {
                let a = result[position];
                let b = result[position + length / 2] * w;
                result[position] = a + b;
                result[position + length / 2] = a - b;
                w = w * radius;
            }
        }
    }
    let scale = T::from(n).unwrap().recip();
    result.iter().map(|x| *x * scale).collect()
}
/// Computes the Inverse Fast Fourier Transform of an one-dimensional, real-valued signal.
/// TODO: Fix the function; currently fails to compute the correct result
pub fn irfft<T>(input: &[Complex<T>], plan: &FftPlan) -> Vec<T>
where
    T: Float + FloatConst,
    Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
{
    let n = input.len();
    let mut result = vec![Complex::new(T::zero(), T::zero()); n];

    for position in plan.clone().into_iter() {
        result.push(input[position]);
    }
    // for res in result.clone() {
    //     if res.im() > T::zero() {
    //         result.push(res.conj());
    //     }
    // }
    // segment length
    let mut segment: usize = 1;
    while segment < n {
        segment <<= 1;
        // compute the angle of the complex number
        let angle = fft_angle::<T>(segment).neg();
        // compute the radius of the complex number (length)
        let radius = Complex::new(T::cos(angle), T::sin(angle));
        for start in (0..n).step_by(segment) {
            let mut w = Complex::new(T::one(), T::zero());
            for position in start..(start + segment / 2) {
                let a = result[position];
                let b = result[position + segment / 2] * w;
                result[position] = a + b;
                result[position + segment / 2] = a - b;
                w *= radius;
            }
        }
    }
    let scale = T::from(n).unwrap().recip();
    result.iter().map(|x| x.re() * scale).collect()
}

#[doc(hidden)]
/// Generates a permutation for the Fast Fourier Transform.
pub(crate) fn fft_permutation(length: usize) -> Vec<usize> {
    let mut result = Vec::new();
    result.reserve_exact(length);
    for i in 0..length {
        result.push(i);
    }
    let mut reverse = 0_usize;
    let mut position = 1_usize;
    while position < length {
        let mut bit = length >> 1;
        while bit & reverse != 0 {
            reverse ^= bit;
            bit >>= 1;
        }
        reverse ^= bit;
        // This is equivalent to adding 1 to a reversed number
        if position < reverse {
            // Only swap each element once
            result.swap(position, reverse);
        }
        position += 1;
    }
    result
}
