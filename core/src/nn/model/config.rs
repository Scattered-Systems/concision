/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait RawConfig {}

pub struct ConfigBase {
    pub id: usize,
    pub name: String,
    pub description: String,

    _children: Vec<Box<dyn RawConfig>>,
}
