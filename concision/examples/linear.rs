/*
    Appellation: linear <example>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision as cnc;

use cnc::func::Sigmoid;
use cnc::linear::{Config, Features, Linear};
use cnc::{linarr, Predict, Result};

fn tracing() {
    use tracing::Level::DEBUG;
    use tracing_subscriber::fmt::time;

    tracing_subscriber::fmt()
        .compact()
        .with_ansi(true)
        .with_level(true)
        .with_max_level(DEBUG)
        .with_target(false)
        .with_timer(time::uptime())
        .init();
}

fn main() -> Result<()> {
    tracing();
    tracing::info!("Starting linear model example");

    let (samples, dmodel, features) = (20, 5, 3);
    let features = Features::new(features, dmodel);
    let config = Config::new("example", features).biased();
    let data = linarr::<f64, ndarray::Ix2>((samples, dmodel)).unwrap();

    let model: Linear<f64> = Linear::std(config).uniform();

    let y = model.activate(&data, Sigmoid::sigmoid).unwrap();
    assert_eq!(y.dim(), (samples, features));
    println!("Predictions: {:?}", y);

    Ok(())
}
