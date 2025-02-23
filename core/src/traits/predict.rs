/*
   Appellation: predict <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub type BoxPredict<T = ndarray::Array2<f64>, O = T> = Box<dyn Predict<T, Output = O>>;

/// [Forward] describes an object capable of forward propagation; that is, it can
/// take an input and produce an output.
///
pub trait Forward<T> {
    type Output;

    fn forward(&self, args: &T) -> Self::Output;
}

/// [ForwardIter] describes any iterators whose elements implement [Forward].
/// This trait is typically used in deep neural networks who need to forward propagate
/// across a number of layers.
pub trait ForwardIter<T> {
    type Item: Forward<T, Output = T>;

    fn forward_iter(self, args: &T) -> <Self::Item as Forward<T>>::Output;
}

/// The [Predict] is a generalized implementation of the [Forward] trait equipped with
/// additional error handling capabilities.
pub trait Predict<T = ndarray::Array2<f64>> {
    type Output;

    fn predict(&self, args: &T) -> crate::Result<Self::Output>;
}
/*
 ********* Implementations *********
*/
impl<U, M> Forward<U> for M
where
    M: Predict<U>,
{
    type Output = Option<M::Output>;

    fn forward(&self, args: &U) -> Self::Output {
        self.predict(args).ok()
    }
}

impl<I, M, T> ForwardIter<T> for I
where
    I: IntoIterator<Item = M>,
    M: Forward<T, Output = T>,
    T: Clone,
{
    type Item = M;

    fn forward_iter(self, args: &T) -> M::Output {
        self.into_iter()
            .fold(args.clone(), |acc, m| m.forward(&acc))
    }
}
#[cfg(feature = "alloc")]
impl<U, V> Predict<U> for Box<dyn Predict<U, Output = V>> {
    type Output = V;

    fn predict(&self, args: &U) -> crate::Result<Self::Output> {
        self.as_ref().predict(args)
    }
}

impl<S, T> Predict<T> for Option<S>
where
    S: Predict<T, Output = T>,
    T: Clone,
{
    type Output = T;

    fn predict(&self, args: &T) -> crate::Result<Self::Output> {
        match self {
            Some(s) => s.predict(args),
            None => Ok(args.clone()),
        }
    }
}
