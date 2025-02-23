# concision (cnc)

[![license](https://img.shields.io/crates/l/concision.svg)](https://choosealicense.com/licenses/apache-2.0/)
[![crates.io](https://img.shields.io/crates/v/concision.svg)](https://crates.io/crates/concision)
[![docs.rs](https://docs.rs/concision/badge.svg)](https://docs.rs/concision)

[![clippy](https://github.com/FL03/concision/actions/workflows/clippy.yml/badge.svg)](https://github.com/FL03/concision/actions/workflows/clippy.yml)
[![rust](https://github.com/FL03/concision/actions/workflows/rust.yml/badge.svg)](https://github.com/FL03/concision/actions/workflows/rust.yml)

***

_**Warning: The library is currently in the early stages of development and is not yet ready for production use.**_

Concision is designed to be a complete toolkit for building machine learning models in Rust.

Concision is a machine learning library for building powerful models in Rust prioritizing ease-of-use, efficiency, and flexability. The library is built to make use of the both the upcoming `autodiff` experimental feature and increased support for generics in the 2024 edition of Rust.

## Getting Started

### Building from the source

Start by cloning the repository

```bash
git clone https://github.com/FL03/concision.git
cd concision
```

```bash
cargo build --features full -r --workspace
```

### Testing the crate

```bash
cargo test --workspace -F full
```

## Usage

### Example: Linear Model (biased)

```rust
    extern crate concision as cnc;

    use cnc::prelude::{linarr, Linear, Result, Sigmoid};
    use ndarray::Ix2;

    fn main() -> Result<()> {
        tracing_subscriber::fmt::init();
        tracing::info!("Starting linear model example");

        let (samples, d_in, d_out) = (20, 5, 3);
        let data = linarr::<f64, Ix2>((samples, d_in)).unwrap();

        let model = Linear::<f64>::from_features(d_in, d_out).uniform();
        // let model = Linear::<f64, cnc::linear::Unbiased>::from_features(d_in, d_out).uniform();

        assert!(model.is_biased());

        let y = model.activate(&data, Sigmoid::sigmoid).unwrap();
        assert_eq!(y.dim(), (samples, d_out));
        println!("Predictions:\n{:?}", &y);

        Ok(())
    }
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

* [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
* [MIT](https://choosealicense.com/licenses/mit/)
