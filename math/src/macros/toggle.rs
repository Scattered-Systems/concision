/*
    Appellation: toggle <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[macro_export]
macro_rules! toggle {
    ($vis:vis enum {$($name:ident),* $(,)?}) => {
        $(toggle!(@impl $vis enum $name);)*
    };

    (@impl $vis:vis enum $name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde_derive::Deserialize, serde_derive::Serialize))]
        $vis enum $name {}
    };
}
