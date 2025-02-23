/*
   Appellation: concision-gnn <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Structured State Space Sequential Models (S4)
//!
//!
//! ## Resources
//!
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_name = "concision_s4"]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision_core as concision;

pub mod prelude {}
