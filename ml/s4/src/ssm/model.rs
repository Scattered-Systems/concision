/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMConfig;
use faer::prelude::{FaerMat, IntoFaer, SolverCore};
use faer::IntoNdarray;
use faer_core::zip::ViewMut;
use faer_core::{ComplexField, Conjugate, SimpleEntity};
use ndarray::prelude::{s, Array1, Array2, ArrayView1, NdFloat};
use ndarray::IntoDimension;
// use ndarray_linalg::solve::Inverse;
// use ndarray_linalg::types::Lapack;
use num::{Float, ToPrimitive};

pub type ScanFn<O, S, T> = Box<dyn Fn(&mut S, T) -> Option<O>>;

pub fn scanner<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    u: &Array2<T>,
    x0: &Array1<T>,
) -> Array2<T>
where
    T: NdFloat,
{
    let step = |xs: &mut Array1<T>, us: ArrayView1<T>| {
        let x1 = a.dot(xs) + b.dot(&us);
        let y1 = c.dot(&x1);
        Some(y1)
    };
    let scan = u.outer_iter().scan(x0.clone(), step).collect::<Vec<_>>();
    let shape = [scan.len(), scan[0].len()];
    let mut res = Array2::<T>::zeros(shape.into_dimension());
    for (i, s) in scan.iter().enumerate() {
        res.slice_mut(s![i, ..]).assign(s);
    }
    res
}

pub struct SSM<T = f64> {
    config: SSMConfig,
    pub a: Array2<T>,
    pub b: Array2<T>,
    pub c: Array2<T>,
    pub d: Array2<T>,
}

impl<T> SSM<T>
where
    T: Float,
{
    pub fn create(config: SSMConfig) -> Self {
        let features = config.features();
        let a = Array2::<T>::zeros((features, features));
        let b = Array2::<T>::zeros((features, 1));
        let c = Array2::<T>::zeros((1, features));
        let d = Array2::<T>::zeros((1, 1));
        Self { config, a, b, c, d }
    }
}

impl<T> SSM<T>
where
    T: NdFloat + Conjugate + SimpleEntity,
    <T as Conjugate>::Canonical: ComplexField + SimpleEntity + ToPrimitive,
{
    pub fn discretize(&mut self, step: T) -> anyhow::Result<()> {
        let ds = step / T::from(2).unwrap();
        let eye = Array2::<T>::eye(self.config.features());
        let bl = &eye - &self.a * ds;
        let be = {
            let mut tmp = bl.view().into_faer().qr().inverse();
            let arr = &tmp.view_mut().into_ndarray();
            arr.mapv(|i| T::from(i).unwrap())
        };
        let ab = &be.dot(&(&eye + &self.a * ds));
        let bb = (&self.b * ds).dot(&self.b.t());

        Ok(())
    }
}

// impl SSM<f64> {

//     pub fn descretize(&mut self, step: f64) -> anyhow::Result<()> {
//         let ds = step / 2.0;
//         let eye = Array2::<f64>::eye(self.config.features());
//         let bl = (&eye - &self.a * ds).inv()?;
//         // let ab = &bl | (&eye + &self.a * ds);
//         // let bb = &self.b * ds | self.b;

//         Ok(())
//     }
// }
