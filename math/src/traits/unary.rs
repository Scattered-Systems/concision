/*
    Appellation: unary <traits>
    Contrib: @FL03
*/
use ndarray::{Array, ArrayBase, Data, Dimension};
use num::complex::{Complex, ComplexFloat};
use num::traits::Signed;

unary!(
    Abs::abs(self),
    Cos::cos(self),
    Cosh::cosh(self),
    Exp::exp(self),
    Sine::sin(self),
    Sinh::sinh(self),
    Squared::sqrd(self),
    SquareRoot::sqrt(self)
);

/*
 ********* Implementations *********
*/

macro_rules! unary_impl {
    ($name:ident::$method:ident<[$($T:ty),* $(,)?]>) => {
        unary_impl!(@loop $name::$method<[$($T),*]>);
    };
    ($($name:ident::$method:ident<$T:ty$(, Output = $O:ty)?>),* $(,)?) => {
        $(unary_impl!(@impl $name::$method<$T$(, Output = $O>)?);)*
    };
    ($($name:ident::$method:ident<$T:ty, Output = $O:ty>),* $(,)?) => {
        $(unary_impl!(@impl $name::$method<$T, Output = $O>);)*
    };
    (@loop $name:ident::$method:ident<[$($T:ty),* $(,)?]>) => {
        $(unary_impl!(@impl $name::$method<$T>);)*
    };
    (@impl $name:ident::$method:ident<$T:ty>) => {
        unary_impl!(@impl $name::$method<$T, Output = $T>);
    };
    (@impl $name:ident::$method:ident<$T:ty, Output = $O:ty>) => {
        impl $name for $T {
            type Output = $O;

            fn $method(self) -> Self::Output {
                <$T>::$method(self)
            }
        }
    };
}

macro_rules! unary_impls {
    ($($name:ident::$method:ident<[$($T:ty),* $(,)?]>),* $(,)?) => {
        $(unary_impl!(@loop $name::$method<[$($T),*]>);)*
    };
}

unary_impls!(
    Abs::abs<[f32, f64]>,
    Cosh::cosh<[f32, f64, Complex<f32>, Complex<f64>]>,
    Cos::cos<[f32, f64, Complex<f32>, Complex<f64>]>,
    Exp::exp<[f32, f64, Complex<f32>, Complex<f64>]>,
    Sinh::sinh<[f32, f64, Complex<f32>, Complex<f64>]>,
    Sine::sin<[f32, f64, Complex<f32>, Complex<f64>]>,
    SquareRoot::sqrt<[f32, f64]>
);

impl<A, S, D> Abs for ArrayBase<S, D>
where
    A: Clone + Signed,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn abs(self) -> Self::Output {
        self.mapv(|x| x.abs())
    }
}

impl<'a, A, S, D> Abs for &'a ArrayBase<S, D>
where
    A: Clone + Signed,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn abs(self) -> Self::Output {
        self.mapv(|x| x.abs())
    }
}

impl<A> Squared for A
where
    A: Clone + core::ops::Mul<Output = A>,
{
    type Output = A;

    fn sqrd(self) -> Self::Output {
        self.clone() * self
    }
}

impl<A> SquareRoot for Complex<A>
where
    Complex<A>: ComplexFloat<Real = A>,
{
    type Output = Self;

    fn sqrt(self) -> Self::Output {
        ComplexFloat::sqrt(self)
    }
}

impl<A, B, S, D> SquareRoot for ArrayBase<S, D>
where
    A: Clone + SquareRoot<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn sqrt(self) -> Self::Output {
        self.mapv(|x| x.sqrt())
    }
}

impl<A, B, S, D> Exp for ArrayBase<S, D>
where
    A: Clone + Exp<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn exp(self) -> Self::Output {
        self.mapv(|x| x.exp())
    }
}

impl<'a, A, S, D> Exp for &'a ArrayBase<S, D>
where
    A: Clone + ComplexFloat,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn exp(self) -> Self::Output {
        self.mapv(|x| x.exp())
    }
}
