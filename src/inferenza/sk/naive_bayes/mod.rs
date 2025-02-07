mod gaussian_nb;

use ndarray::{Array2, Array1};

// Trait for naive bayes algorithms
pub trait NaiveBayesAlgorithm<f32> {
    // Inference method for NaiveBayes algorithms
    fn predict(&self, input: Array2<f32>) -> Array1<f32>;
}