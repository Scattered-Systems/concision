/*
   Appellation: modules <traits::nn>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::Forward;

pub type ModuleDyn<T, C, P> = Box<dyn Module<CSpace = C, Elem = T, Params = P>>;

/// A `Module` defines any object that may be used as a layer in a neural network.
/// [Config](Module::Config) contains all of the hyperparameters for the model.
/// [Params](Module::Params) refers to an object used to store the various learnable parameters.
pub trait Module {
    type CSpace; // Config Space
    type Elem;
    type Params;

    fn config(&self) -> &Self::CSpace;

    fn params(&self) -> &Self::Params;

    fn params_mut(&mut self) -> &mut Self::Params;
}

pub trait Layer<T>: Module + Forward<T> {}

/*
 ************* Implementations *************
*/

impl<T, M> Layer<T> for M where M: Module + Forward<T> {}
