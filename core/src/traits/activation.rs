/*
    Appellation: activation <module>
    Contrib: @FL03
*/

pub trait ActivationFn<T> {
    fn activate(&self, x: T) -> T;
}

pub trait Neuron<T> {
    type Rho: ActivationFn<T>;

    fn compute(&self, input: T) -> T;
    fn learn(&mut self, target: T, eta: T);
}

impl<T> ActivationFn<T> for fn(T) -> T {
    fn activate(&self, x: T) -> T {
        self(x)
    }
}
