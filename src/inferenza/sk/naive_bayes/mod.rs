mod gaussian_nb;

use ndarray::{Array2, Array1};

// Trait for naive bayes algorithms
pub trait NaiveBayesAlgorithm<T> {
    // Inference method for NaiveBayes algorithms
    fn.predict(&self, input: Array2<T>) -> Array1<u32>;
}