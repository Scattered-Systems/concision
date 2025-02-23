/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::{TypeTag, toggle};

pub trait ParamMode: TypeTag {
    const BIASED: bool = false;

    fn is_biased(&self) -> bool {
        core::any::TypeId::of::<Self>() == core::any::TypeId::of::<Biased>()
    }

    private!();
}

/*
 ************* Implementations *************
*/
macro_rules! mode {
    {$($T:ident: $opt:expr),* $(,)?} => {
        $(mode!(@impl $T: $opt);)*
    };
    (@impl $T:ident: $opt:expr) => {
        concision::toggle!(pub enum {$T});

        impl ParamMode for $T {
            const BIASED: bool = $opt;

            fn is_biased(&self) -> bool {
                $opt
            }

            seal!();
        }
    };

}

mode! {
    Biased: true,
    Unbiased: false,
}
