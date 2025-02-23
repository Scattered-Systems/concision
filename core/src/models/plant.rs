/*
    Appellation: plant <module>
    Contrib: @FL03
*/
use crate::Perceptron;

pub struct PlantModel {
    pub input: [Perceptron; 3],
    pub hidden: [[Perceptron; 3]; 3],
    pub output: Perceptron,
}
