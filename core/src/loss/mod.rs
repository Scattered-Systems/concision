/*
    Appellation: loss <module>
    Contrib: @FL03
*/
//! This module is responsible for implementing various loss functions used in machine
//! learning.
#[doc(inline)]
pub use self::prelude::*;

pub mod means;

pub(crate) mod prelude {
    pub use super::means::*;
}

#[cfg(test)]
mod test {
    use super::MeanAbsoluteError;
    use ndarray::array;

    #[test]
    fn test_mae_loss() {
        let pred = array![1.0, 2.0, 3.0, 4.0];
        let actual = array![1.0, 3.0, 3.5, 4.5];
        assert_eq!(pred.mae(&actual), Some(0.5));
    }
}
