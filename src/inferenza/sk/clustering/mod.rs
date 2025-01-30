mod kmeans;

use ndarray::{Array2, Array1};

/// Trait for clustering algorithms
pub trait ClusteringAlgorithm<T> {
    /// Inference method for clustering algorithms
    fn predict(&self, input: Array2<T>) -> Array1<u32>;
}
