/*
    Appellation: basic <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use cnc::ops::pad;
use ndarray::prelude::Array;

fn add(x: f64, y: f64) -> f64 {
    x + y
}

fn main() -> anyhow::Result<()> {
    println!("Welcome to concision!");
    dbg!(add(1.0, 2.0));
    dbg!(add(3.0, 4.0));
    Ok(())
}

pub fn sample_padding(samples: usize) -> anyhow::Result<()> {
    let arr = Array::range(0.0, (samples * samples) as f64, 1.0);
    let arr = arr.to_shape((samples, samples))?;
    let padded = pad(&arr, &[[0, 8]], 0.0.into());
    println!("{:?}", &padded);
    Ok(())
}
