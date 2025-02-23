/*
    Appellation: checks <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use core::fmt;

/// A function helper for testing that some result is ok
pub fn assert_ok<T, E>(res: Result<T, E>) -> T
where
    E: fmt::Debug,
{
    assert!(res.is_ok(), "{:?}", res.err());
    res.unwrap()
}
