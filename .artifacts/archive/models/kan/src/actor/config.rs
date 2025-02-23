/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::Config;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ActorConfig;

impl Config for ActorConfig {}

impl Default for ActorConfig {
    fn default() -> Self {
        Self
    }
}
