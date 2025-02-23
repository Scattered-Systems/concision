/*
    Appellation: build <build>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_std)");
}
