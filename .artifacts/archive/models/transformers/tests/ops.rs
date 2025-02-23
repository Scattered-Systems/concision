/*
    Appellation: ops <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_transformer as transformer;
extern crate ndarray as nd;

use concision::linarr;
use nd::prelude::*;
use transformer::ops::*;

pub const HEADS: usize = 2;
pub const ORDER: nd::Order = nd::Order::RowMajor;

#[test]
fn test_merge() {
    let shape = (3, 4, 5);
    let dout = (4, 15);
    let arr = linarr::<f64, Ix3>(shape.clone()).unwrap();
    let a = arr.merge().unwrap();

    assert_eq!(a.dim(), dout);
    assert_eq!(a, utils::merge3(&arr).unwrap());
}

#[test]
fn test_merge_batch() {
    let shape = (2, 3, 4, 5);
    let dout = (2, 4, 15);
    let arr = linarr::<f64, Ix4>(shape).unwrap();
    let a = arr.merge().unwrap();

    assert_eq!(a.dim(), dout);
    assert_eq!(a, utils::merge4(&arr).unwrap());
}

#[test]
fn test_split() {
    let heads = 2;
    let shape = (4, 6);
    let arr = linarr::<f64, Ix2>(shape).unwrap();
    let a = arr.split(heads).unwrap();

    assert_eq!(a.dim(), (heads, 4, 3));
    assert_eq!(a, utils::split_heads(&arr, heads).unwrap());
}

#[test]
fn test_split_batch() {
    let heads = 2;
    let shape = (3, 4, 6);
    let arr = linarr::<f64, Ix3>(shape).unwrap();
    let a = arr.split(heads).unwrap();

    assert_eq!(a.dim(), (3, heads, 4, 3));
    assert_eq!(a, utils::split_batch(&arr, heads).unwrap());
}

#[test]
fn reshape_ops() {
    let shape = (2, 4, 6);
    let data = linarr::<f64, Ix3>(shape).unwrap();

    let a = data.split(HEADS).unwrap();
    assert_eq!(a.dim(), (2, HEADS, 4, 3));
    let b = a.merge().unwrap();
    assert_eq!(b.dim(), shape);
    // verify that doing the ops consecutively is the identity
    assert_eq!(b, data);
}

#[allow(dead_code)]
pub(crate) mod utils {
    use concision::NdResult;
    use ndarray::*;

    pub fn merge3<T>(heads: &Array3<T>) -> NdResult<Array2<T>>
    where
        T: Clone,
    {
        let (n, seq, query) = heads.dim();
        let shape = (seq, n * query);
        let mut tmp = heads.clone();
        // swap the head and sequence axes
        tmp.swap_axes(0, 1);
        // reshape the qkv matrix into a 2d array
        tmp.to_shape((shape, super::ORDER)).map(|x| x.to_owned())
    }

    pub fn merge4<T>(heads: &Array4<T>) -> NdResult<Array3<T>>
    where
        T: Clone,
    {
        let (batch, n, seq, query) = heads.dim();
        let shape = (batch, seq, n * query);
        let mut tmp = heads.clone();
        // swap the head and sequence axes
        tmp.swap_axes(1, 2);
        // reshape the qkv matrix into a 2d array
        tmp.to_shape((shape, super::ORDER)).map(|x| x.to_owned())
    }

    pub fn split_heads<T>(param: &Array2<T>, h: usize) -> NdResult<Array3<T>>
    where
        T: Clone,
    {
        let dim = param.shape().last().unwrap() / h;
        // reshape the qkv matrix into a 3d array
        let mut res = param.to_shape((param.shape()[0], h, dim))?;
        // swap the sequence and head axes
        res.swap_axes(0, 1);
        Ok(res.to_owned())
    }

    pub fn split_batch<T>(param: &Array3<T>, h: usize) -> NdResult<Array4<T>>
    where
        T: Clone,
    {
        let dim = param.shape().last().unwrap() / h;
        // reshape the qkv matrix into a 3d array
        let mut res = param.to_shape((param.shape()[0], param.shape()[1], h, dim))?;
        // swap the sequence and head axes
        res.swap_axes(1, 2);
        Ok(res.to_owned())
    }
}
