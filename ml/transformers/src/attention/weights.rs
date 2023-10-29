/*
   Appellation: weights <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::ops::Split;
use super::params::QKV;

use ndarray::prelude::{Array, Array2, Array3};
use ndarray::{IntoDimension, Ix2};
use num::Float;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use std::ops;


pub type WeightTensor<T = f64> = Array<T, Ix2>; // (seq, model)

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Weight<T = f64> where T: Float {
    dim: Ix2,
    pub key: Array2<T>,
    pub query: Array2<T>,
    pub value: Array2<T>,
}

impl<T: Float> Weight<T> {
    pub fn new<D>(dim: D) -> Self
    where
        D: IntoDimension<Dim = Ix2>,
    {
        let dim = dim.into_dimension();
        let arr = Array2::ones(dim);
        Self {
            dim,
            key: arr.clone(),
            query: arr.clone(),
            value: arr,
        }
    }

    pub fn dim(&self) -> Ix2 {
        self.dim
    }

    pub fn qkv(&self) -> (Array2<T>, Array2<T>, Array2<T>) {
        self.clone().into()
    }
}

impl std::fmt::Display for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

impl<D, T> From<D> for Weight<T>
where
    D: IntoDimension<Dim = Ix2>,
    T: Float
{
    fn from(dim: D) -> Self {
        let dim = dim.into_dimension();
        let arr = Array2::ones(dim);
        Self {
            dim,
            key: arr.clone(),
            query: arr.clone(),
            value: arr,
        }
    }
}

impl<T: Float> From<Weight<T>> for (Array2<T>, Array2<T>, Array2<T>) {
    fn from(context: Weight<T>) -> Self {
        (context.key, context.query, context.value)
    }
}

impl<T: Float> Split<(Array3<T>, Array3<T>, Array3<T>)> for Weight<T> {
    type Error = ndarray::ShapeError;

    fn split(&self, heads: usize) -> Result<(Array3<T>, Array3<T>, Array3<T>), Self::Error> {
        let (key, query, value) = self.qkv();
        Ok((key.split(heads)?, query.split(heads)?, value.split(heads)?))
    }
}

impl<T: Float> ops::Index<QKV> for Weight<T> {
    type Output = Array2<T>;

    fn index(&self, index: QKV) -> &Self::Output {
        use QKV::*;
        match index {
            Key => &self.key,
            Query => &self.query,
            Value => &self.value,
        }
    }
}

impl<T: Float> ops::IndexMut<QKV> for Weight<T> {
    fn index_mut(&mut self, index: QKV) -> &mut Self::Output {
        use QKV::*;
        match index {
            Key => &mut self.key,
            Query => &mut self.query,
            Value => &mut self.value,
        }
    }
}

impl<T: Float + 'static> ops::Mul<Array2<T>> for Weight<T> {
    type Output = Self;

    fn mul(self, rhs: Array2<T>) -> Self::Output {
        let mut ctx = self.clone();
        ctx.key = ctx.key.dot(&rhs);
        ctx.query = ctx.query.dot(&rhs);
        ctx.value = ctx.value.dot(&rhs);
        ctx
    }
}

impl<T: Float + 'static> ops::Mul<&Array2<T>> for Weight<T> {
    type Output = Self;

    fn mul(self, rhs: &Array2<T>) -> Self::Output {
        let mut ctx = self.clone();
        ctx.key = ctx.key.dot(rhs);
        ctx.query = ctx.query.dot(rhs);
        ctx.value = ctx.value.dot(rhs);
        ctx
    }
}

impl<T: Float + 'static> ops::Mul<&Array2<T>> for &Weight<T> {
    type Output = Weight<T>;

    fn mul(self, rhs: &Array2<T>) -> Self::Output {
        let mut ctx = self.clone();
        ctx.key = ctx.key.dot(rhs);
        ctx.query = ctx.query.dot(rhs);
        ctx.value = ctx.value.dot(rhs);
        ctx
    }
}

impl<T: Float + 'static> ops::MulAssign<Array2<T>> for Weight<T> {
    fn mul_assign(&mut self, rhs: Array2<T>) {
        for qkv in QKV::iter() {
            self[qkv] = self[qkv].dot(&rhs);
        }
    }
}

impl<T: Float + 'static> ops::MulAssign<&Array2<T>> for Weight<T> {
    fn mul_assign(&mut self, rhs: &Array2<T>) {
        for qkv in QKV::iter() {
            self[qkv] = self[qkv].dot(rhs);
        }
    }
}

impl<T: Float + 'static> ops::MulAssign<&Array2<T>> for &mut Weight<T> {
    fn mul_assign(&mut self, rhs: &Array2<T>) {
        for qkv in QKV::iter() {
            self[qkv] = self[qkv].dot(rhs);
        }
    }
}
