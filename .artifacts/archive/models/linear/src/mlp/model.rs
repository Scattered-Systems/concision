/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::{ModelError, Predict};
use core::marker::PhantomData;

// #92: Define the Multi-Layer Perceptron (MLP) model
/// A multi-layer perceptron (MLP) model.
pub struct Mlp<A, I, O>
where
    I: Predict<A>,
    O: Predict<I::Output>,
{
    input: I,
    hidden: Vec<Box<dyn Predict<I::Output, Output = I::Output>>>,
    output: O,
    _dtype: PhantomData<A>,
}

impl<A, I, O> Predict<A> for Mlp<A, I, O>
where
    I: Predict<A>,
    O: Predict<I::Output>,
{
    type Output = O::Output;

    fn predict(&self, input: &A) -> Result<Self::Output, ModelError> {
        let mut hidden = self.input.predict(input)?;
        for layer in &self.hidden {
            hidden = layer.predict(&hidden)?;
        }
        self.output.predict(&hidden)
    }
}
