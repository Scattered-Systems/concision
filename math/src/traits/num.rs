/*
   Appellation: num <traits>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, Dimension};
use num::complex::Complex;
use num::{Float, Num, Signed, Zero};

pub trait ComplexNum<T = f64> {
    type Real: Sized;
}
/// Trait for converting a type into a complex number.
pub trait AsComplex<T> {
    type Complex: ComplexNum<T>;

    fn as_complex(&self, real: bool) -> Self::Complex;

    fn as_re(&self) -> Self::Complex {
        self.as_complex(true)
    }

    fn as_im(&self) -> Self::Complex {
        self.as_complex(false)
    }
}
/// Trait for converting a type into a complex number.
pub trait IntoComplex<T> {
    type Complex: ComplexNum<T>;

    fn into_complex(self, real: bool) -> Self::Complex
    where
        Self: Sized;

    fn into_re(self) -> Self::Complex
    where
        Self: Sized,
    {
        self.into_complex(true)
    }

    fn into_im(self) -> Self::Complex
    where
        Self: Sized,
    {
        self.into_complex(false)
    }
}

pub trait Conjugate {
    type Output;

    fn conj(&self) -> Self::Output;
}

pub trait FloorDiv<Rhs = Self> {
    type Output;

    fn floor_div(self, rhs: Rhs) -> Self::Output;
}

pub trait RoundTo {
    fn round_to(&self, places: usize) -> Self;
}

/*
 ********* Implementations *********
*/
impl<T> ComplexNum<T> for Complex<T>
where
    T: Num,
{
    type Real = T;
}

impl<T> ComplexNum<T> for T
where
    T: Num,
{
    type Real = T;
}

impl<T> AsComplex<T> for T
where
    T: Clone + Num,
{
    type Complex = Complex<T>;

    fn as_complex(&self, real: bool) -> Complex<T> {
        match real {
            true => Complex::new(self.clone(), Self::zero()),
            false => Complex::new(Self::zero(), self.clone()),
        }
    }
}

impl<T> IntoComplex<T> for T
where
    T: Num,
{
    type Complex = Complex<T>;

    fn into_complex(self, real: bool) -> Self::Complex
    where
        Self: Sized,
    {
        match real {
            true => Complex::new(self, T::zero()),
            false => Complex::new(T::zero(), self),
        }
    }
}

impl<T> FloorDiv for T
where
    T: Copy + Num,
{
    type Output = T;

    fn floor_div(self, rhs: Self) -> Self::Output {
        crate::floor_div(self, rhs)
    }
}

impl<T> RoundTo for T
where
    T: Float,
{
    fn round_to(&self, places: usize) -> Self {
        crate::round_to(*self, places)
    }
}

/*
 ************* macro implementations *************
*/

macro_rules! impl_conj {
    ($($t:ident<$res:ident>),*) => {
        $(
            impl_conj!(@impl $t<$res>);
        )*
    };
    (@impl $t:ident<$res:ident>) => {
        impl Conjugate for $t {
            type Output = $res<$t>;

            fn conj(&self) -> Self::Output {
                Complex { re: *self, im: -$t::zero() }
            }
        }
    };
}

impl_conj!(f32<Complex>, f64<Complex>);

impl<T> Conjugate for Complex<T>
where
    T: Clone + Signed,
{
    type Output = Complex<T>;

    fn conj(&self) -> Self {
        Complex::<T>::conj(self)
    }
}

impl<T, D> Conjugate for Array<T, D>
where
    D: Dimension,
    T: Clone + num::complex::ComplexFloat,
{
    type Output = Array<T, D>;
    fn conj(&self) -> Self::Output {
        self.mapv(|x| x.conj())
    }
}
