/*
    Appellation: models <module>
    Contrib: @FL03
*/

pub mod plant;

pub(crate) mod prelude {}

/// A simple Plant AI model that uses barycentric coordinates as weights
/// for three basis functions.
pub struct PlantModel {
    alpha: f64,
    beta: f64,
    gamma: f64,
}

impl PlantModel {
    /// Creates a new Plant instance with equal initial weights (1/3 each).
    pub fn new() -> Self {
        Self {
            alpha: 1.0 / 3.0,
            beta: 1.0 / 3.0,
            gamma: 1.0 / 3.0,
        }
    }

    /// Basis function 1: f1(x) = 2x
    fn f1(&self, x: f64) -> f64 {
        2.0 * x
    }

    /// Basis function 2: f2(x) = x + 1
    fn f2(&self, x: f64) -> f64 {
        x + 1.0
    }

    /// Basis function 3: f3(x) = -x
    fn f3(&self, x: f64) -> f64 {
        -x
    }

    /// Computes the output of the plant for a given input x.
    /// Normalizes the barycentric coordinates before computation.
    pub fn compute(&self, x: f64) -> f64 {
        let sum = self.alpha + self.beta + self.gamma;
        let a = self.alpha / sum;
        let b = self.beta / sum;
        let c = self.gamma / sum;
        a * self.f1(x) + b * self.f2(x) + c * self.f3(x)
    }

    /// Updates the barycentric coordinates based on the error between
    /// the target output y and the predicted output, using learning rate eta.
    pub fn learn(&mut self, x: f64, y: f64, eta: f64) {
        let predicted = self.compute(x);
        let error = y - predicted;
        self.alpha += eta * error * self.f1(x);
        self.beta += eta * error * self.f2(x);
        self.gamma += eta * error * self.f3(x);
        // Normalize the weights to ensure they sum to 1
        let sum = self.alpha + self.beta + self.gamma;
        self.alpha /= sum;
        self.beta /= sum;
        self.gamma /= sum;
    }
}
