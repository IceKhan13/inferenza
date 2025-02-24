mod clustering;
mod supervised;

use ndarray::Array2;

/// Trait for regression algorithms
pub trait Regressor<T> {
    /// Inference method for clustering algorithms
    fn predict(&self, input: Array2<T>) -> Array2<T>;
}
