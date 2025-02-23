/*
    Appellation: serde <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "serde")]

use crate::params::{Parameter, ParamsBase};
use core::marker::PhantomData;
use nd::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<'a, A, S, D, K> Deserialize<'a> for ParamsBase<S, D, K>
where
    A: Deserialize<'a>,
    D: Deserialize<'a> + RemoveAxis,
    S: DataOwned<Elem = A>,
    <D as Dimension>::Smaller: Deserialize<'a> + Dimension,
{
    fn deserialize<Der>(deserializer: Der) -> Result<Self, Der::Error>
    where
        Der: Deserializer<'a>,
    {
        let (bias, weights) = Deserialize::deserialize(deserializer)?;
        Ok(Self {
            bias,
            weight: weights,
            _mode: PhantomData,
        })
    }
}

impl<A, S, D, K> Serialize for ParamsBase<S, D, K>
where
    A: Serialize,
    D: RemoveAxis + Serialize,
    S: Data<Elem = A>,
    <D as Dimension>::Smaller: Dimension + Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        (self.bias.as_ref(), self.weights()).serialize(serializer)
    }
}

impl<A, S, D> Serialize for Parameter<S, D>
where
    A: Serialize,
    S: Data<Elem = A>,
    D: RemoveAxis + Serialize,
    <D as Dimension>::Smaller: Dimension + Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        match self {
            Self::Bias(bias) => bias.serialize(serializer),
            Self::Weight(weight) => weight.serialize(serializer),
        }
    }
}
