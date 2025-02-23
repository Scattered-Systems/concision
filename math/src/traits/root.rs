/*
    Appellation: arithmetic <module>
    Contrib: @FL03
*/
use num::integer::Roots;
use num::traits::FromPrimitive;

pub trait Root {
    type Output;

    fn nth_root(&self, n: u32) -> Self::Output;

    fn sqrt(&self) -> Self::Output {
        self.nth_root(2)
    }

    fn cbrt(&self) -> Self::Output {
        self.nth_root(3)
    }
}

macro_rules! impl_root {
    (float $($T:ty),* $(,)?) => {
        $(
            impl_root!(@float $T);
        )*
    };
    ($($T:ty),* $(,)?) => {
        $(
            impl_root!(@impl $T);
        )*
    };

    (@impl $T:ty) => {
        impl Root for $T {
            type Output = $T;

            fn nth_root(&self, n: u32) -> Self::Output {
                Roots::nth_root(self, n)
            }
        }
    };
    (@float $T:ty) => {
        impl Root for $T {
            type Output = $T;

            fn nth_root(&self, n: u32) -> Self::Output {
                self.powf(<$T>::from_u32(n).unwrap().recip())
            }
        }
    };
}

impl_root!(float f32, f64);
impl_root! {
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
}
