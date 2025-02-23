/*
    Appellation: dropout <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

/// [Dropout] randomly zeroizes elements with a given probability (`p`).
pub trait DropOut {
    type Output;

    fn dropout(&self, p: f64) -> Self::Output;
}

/// The [Dropout] layer is randomly zeroizes inputs with a given probability (`p`).
/// This regularization technique is often used to prevent overfitting.
///
///
/// ### Config
///
/// - (p) Probability of dropping an element
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Dropout {
    pub(crate) p: f64,
}

/*
 ************* Implementations *************
*/

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self { p }
    }

    pub fn scale(&self) -> f64 {
        (1f64 - self.p).recip()
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

#[cfg(feature = "rand")]
mod impls {
    use super::*;

    use crate::Forward;
    use ndarray::{Array, ArrayBase, DataOwned, Dimension, ScalarOperand};
    use num::traits::Num;

    pub(crate) fn _dropout<S, A, D>(array: &ArrayBase<S, D>, p: f64) -> Array<A, D>
    where
        A: Num + ScalarOperand,
        D: Dimension,
        S: DataOwned<Elem = A>,
    {
        use crate::init::InitializeExt;
        // Create a mask of the same shape as the input array
        let mask: ndarray::Array<bool, D> =
            ndarray::Array::bernoulli(array.dim(), p).expect("Failed to create mask");
        let mask = mask.mapv(|x| if x { A::zero() } else { A::one() });

        // Element-wise multiplication to apply dropout
        array.to_owned() * mask
    }

    impl<A, S, D> DropOut for ArrayBase<S, D>
    where
        A: Num + ScalarOperand,
        D: Dimension,
        S: DataOwned<Elem = A>,
    {
        type Output = Array<A, D>;

        fn dropout(&self, p: f64) -> Self::Output {
            _dropout(self, p)
        }
    }

    impl Dropout {
        pub fn apply<A, S, D>(&self, input: &ArrayBase<S, D>) -> Array<A, D>
        where
            A: Num + ScalarOperand,
            D: Dimension,
            S: DataOwned<Elem = A>,
        {
            _dropout(input, self.p)
        }
    }

    impl<A, S, D> Forward<ArrayBase<S, D>> for Dropout
    where
        A: Num + ScalarOperand,
        D: Dimension,
        S: DataOwned<Elem = A>,
    {
        type Output = Array<A, D>;

        fn forward(&self, input: &ArrayBase<S, D>) -> Self::Output {
            input.dropout(self.p)
        }
    }
}
