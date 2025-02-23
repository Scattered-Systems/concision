/*
    Appellation: traits <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::Biased;

pub trait IsBiased {
    fn is_biased(&self) -> bool;
}

impl<T> IsBiased for T
where
    T: 'static,
{
    fn is_biased(&self) -> bool {
        core::any::TypeId::of::<T>() == core::any::TypeId::of::<Biased>()
    }
}
