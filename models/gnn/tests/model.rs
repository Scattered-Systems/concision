/*
    Appellation: model <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![allow(unused)]
#![cfg(test)]

extern crate concision_core as concision;
extern crate concision_gnn as gnn;

use concision::prelude::{linarr, Forward};

use lazy_static::lazy_static;

const SAMPLES: usize = 20;
const INPUTS: usize = 5;
const OUTPUT: usize = 3;

lazy_static! {
    static ref CONFIG: (usize, usize, usize) = (SAMPLES, INPUTS, OUTPUT);
}

#[test]
fn test_gnn() {
    let (samples, input, output) = CONFIG.clone();
}
