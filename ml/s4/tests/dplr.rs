#[cfg(test)]
extern crate concision_core;
extern crate concision_s4;

use concision_core as core;
use concision_s4 as s4;

use lazy_static::lazy_static;
use ndarray::prelude::*;
use ndarray_linalg::flatten;
use num::complex::{Complex, ComplexFloat};

use core::prelude::{seeded_uniform, AsComplex, Conjugate, Power};
use s4::cmp::kernel::kernel_dplr;
use s4::hippo::dplr::DPLR;
use s4::ops::{discretize, k_conv};
use s4::params::DPLRParams;

const FEATURES: usize = 4;
const RNGKEY: u64 = 1;
const SAMPLES: usize = 16;

lazy_static! {
    static ref SEEDED_C: Array2<f64> = seeded_uniform(RNGKEY, 0.0, 1.0, (1, FEATURES));
    static ref SAMPLE_C: Array2<f64> = array![[0.02185547, 0.20907068, 0.23742378, 0.3723395]];
    static ref SAMPLE_IM: Array2<Complex<f64>> = SAMPLE_C.clone().mapv(AsComplex::as_re);
}

#[test]
// #[ignore = "TODO: fix this test"]
fn test_gen_dplr() {
    let eye = Array2::<f64>::eye(FEATURES);

    let step = (SAMPLES as f64).recip();

    let dplr = DPLR::<f64>::new(FEATURES);
    let (lambda, p, b, _v) = dplr.into();

    println!("{:?}", &p);

    let b2 = b.clone().insert_axis(Axis(1));

    let p2 = p.clone().insert_axis(Axis(1));

    let a = Array::from_diag(&lambda) - p2.dot(&p2.conj().t());

    // let c = {
    //     let tmp = seeded_uniform(RNGKEY, 0.0, 1.0, (1, features));
    //     println!("C:\n\n{:#?}\n", &tmp);
    //     tmp.mapv(AsComplex::as_re)
    // };
    let c = {
        let tmp = array![[0.02185547, 0.20907068, 0.23742378, 0.3723395]];
        println!("C:\n\n{:#?}\n", &tmp);
        tmp.mapv(AsComplex::as_re)
    };

    // TODO: figure out why several of the signs are wrong
    let discrete = {
        let tmp = discretize(&a, &b2, &c, step);
        assert!(tmp.is_ok(), "discretize failed: {:?}", tmp.err().unwrap());
        tmp.unwrap()
    };

    let (ab, bb, cb) = discrete.into();
    //
    let ak = k_conv(&ab, &bb, &cb.conj(), SAMPLES);
    //
    let cc = (&eye - ab.pow(SAMPLES)).conj().t().dot(&flatten(cb));
    //
    let params = DPLRParams::new(lambda, p.clone(), p.clone(), b.clone(), cc);
    //
    let kernal = kernel_dplr::<f64>(&params, step, SAMPLES);
    println!("Kernal: {:?}", kernal.shape());

    let a_real = ak.mapv(|i| i.re());
    let err = (&a_real - &kernal).mapv(|i| i.abs());
    assert!(
        err.mean().unwrap() <= 1e-4,
        "Error: {:?}\nTolerance: {:?}",
        err.mean().unwrap(),
        1e-4
    );
}
