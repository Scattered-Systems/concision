/*
   Appellation: actors <module>
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       Actors are defined to be abstract models of computation and thus represent a collection
       of complex, dynamic computational models equipped with a standard interface enabling for
       compatibility and usability
*/
pub use aggregators::*;
pub use automata::*;
pub use converters::*;
pub use transformers::*;

mod aggregators;
mod automata;
mod converters;
mod transformers;
