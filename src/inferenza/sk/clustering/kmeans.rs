use ndarray::{Array1, Array2};
use crate::onnx;
use crate::inferenza::sk::clustering::ClusteringAlgorithm;
use crate::inferenza::OnnxLoadable;
use crate::inferenza::utils;


/// Kmeans Scikit learn model
#[derive(Debug)]
struct KmeansModel<T> {
    // number of clusters
    n_clusters: u32,
    // cluster centers in form of NDArray
    cluster_centers: Array2<T>,
}

impl OnnxLoadable<KmeansModel<f32>> for KmeansModel<f32> 
{
    /// Loads Kmeans model from ONNX proto model
    fn load_from_onnx_proto(model: onnx::ModelProto) -> KmeansModel<f32> {
        // get model graph
        let graph = model.graph
            .expect("NO graph in the model");

        // read constructor data from model proto
        // get third element (index 2) of initializer array, i.e. cluster centers
        let initializer_centers_data = graph.initializer
            .get(2)
            .expect("No cluster centers in initializer.");
        // get shape from dimension field of initializer to shape cluster centers into NDArray
        let shape = (
            initializer_centers_data.dims[0] as usize,
            initializer_centers_data.dims[1] as usize
        );
        // shape cluster centers into NDArray (2d array, n_features by n_clusters)
        let cluster_centers: Array2<f32> = Array2::from_shape_vec(
            shape, initializer_centers_data.float_data.clone()
        ).unwrap();

        KmeansModel {
            n_clusters: shape.0 as u32,
            cluster_centers,
        }
    }
}

/// Calculates euclidian distance for to NDArrays
/// TODO: check if built-in functions can be used instead.
fn euclidean_distance<T>(a: &Array1<T>, b: &Array1<T>) -> T
where
    T: ndarray::LinalgScalar + num::Float,
{
    assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    let diff = a - b;
    let squared_diff = diff.mapv(|x| x * x);
    let sum_squared_diff = squared_diff.sum();
    sum_squared_diff.sqrt()
}


impl<T> ClusteringAlgorithm<T> for KmeansModel<T> 
where 
    T: ndarray::LinalgScalar + Clone + std::fmt::Debug + num::Float,
{
    /// Runs inference on given input.
    fn predict(&self, input: Array2<T>) -> Array1<u32> {
        // vector for results
        let mut result: Vec<u32> = vec![];

        // interate over input entries and for each calculate distances to each of the cluster centers
        // get minimal distance and store index of this cluster in results vector
        for (i, row) in input.rows().into_iter().enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut min_index: u32 = 0;

            for (i, center) in self.cluster_centers.rows().into_iter().enumerate() {
                let distance = euclidean_distance(&row.to_owned(), &center.to_owned()).to_f64().expect("Cannot cast to float =(");
                if distance < min_distance {
                    min_distance = distance;
                    min_index = i as u32;
                }
            }

            result.push(min_index);
        }   

        Array1::from_vec(result)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans() {
        let model = utils::load_onnx_model_proto("tests/resources/kmeans.onnx")
            .expect("Model has not been loaded.");
        let kmeans = KmeansModel::load_from_onnx_proto(model);
        
        let input = Array2::from_shape_vec(
            (3, 3), vec![
                3.9755416, -9.76483, 9.557824, // 0
                3.9755416, -9.76483, 9.557824, // 0
                -2.466838, 9.029932, 4.4263315 // 1
            ]
        ).unwrap();

        let prediction = kmeans.predict(
            input
        );

        assert_eq!(vec![0, 0, 1], prediction.to_vec());
    } 
}
