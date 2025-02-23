/*
    Appellation: linear <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

impl<'a, T> crate::activate::LinearActivation for &'a T
where
    T: Clone,
{
    type Output = T;

    fn linear(self) -> Self::Output {
        self.clone()
    }
}
