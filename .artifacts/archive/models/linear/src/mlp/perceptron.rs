/*
   Appellation: perceptron <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::prelude::{Activate, ModelError, Module, Predict};
use nd::{ArrayBase, Data, Dimension};

// #91
/// Perceptrons are the fundamental building block of multi-layer perceptrons (MLPs).
/// They are used to model a particular layer within a neural network. Generally speaking,
/// Perceptrons consist of a linear set of parameters and an activation function.
pub struct Perceptron<M, F> {
    module: M,
    rho: F,
}

impl<M, F> Perceptron<M, F>
where
    M: Module,
{
    pub fn new(module: M, rho: F) -> Self {
        Self { module, rho }
    }
}

impl<T, M, F> Activate<T> for Perceptron<M, F>
where
    F: Activate<T>,
{
    type Output = F::Output;

    fn activate(&self, args: T) -> Self::Output {
        self.rho.activate(args)
    }
}

impl<A, S, D, M, F> Predict<ArrayBase<S, D>> for Perceptron<M, F>
where
    D: Dimension,
    S: Data<Elem = A>,
    F: Activate<M::Output>,
    M: Module + Predict<ArrayBase<S, D>>,
{
    type Output = F::Output;

    fn predict(&self, args: &ArrayBase<S, D>) -> Result<Self::Output, ModelError> {
        let res = self.module.predict(args)?;
        Ok(self.rho.activate(res))
    }
}
