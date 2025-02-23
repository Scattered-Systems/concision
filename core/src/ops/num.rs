/*
    Appellation: pow <module>
    Contrib: @FL03
*/

pub trait Power<Rhs = Self> {
    type Output;

    fn pow(self, rhs: Rhs) -> Self::Output;
}

impl<A, B, C> Power<B> for A
where
    A: num::traits::Pow<B, Output = C>,
{
    type Output = C;

    fn pow(self, rhs: B) -> Self::Output {
        self.pow(rhs)
    }
}
