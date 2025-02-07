use ndarray::{Array1, Array2};
use crate::onnx;
use crate::inferenza::sk::naive_bayes::NaiveBayesAlgorithm;
use crate::inferenza::OnnxLoadable;
use crate::inferenza::utils;
use std::f64::consts::PI;
use ndarray::Axis;
use onnx::TensorProto;
use ndarray::stack;


// struct for GaussianNB Scikit learn model, i.e. the minimum data we need from the model to make a prediction on unseen data
#[derive(Debug)] // Q: what does this do?
struct GaussianNBModel {
    classes: Array1<f32>,
    mean: Array2<f32>,
    variance: Array2<f32>,
    log_prior_prob: Array1<f32>,
}

fn convert_tensorproto_to_ndarray(tensor: &TensorProto) -> Array1<f32> {
    let data: Vec<f32> = tensor.float_data.clone(); // Extract data (adjust based on TensorProto type)

    let shape = (
        tensor.dims[0] as usize,
    );

    Array1::from_shape_vec(shape, data).unwrap() // Convert to ndarray
}

fn convert_tensorproto_to_ndarray_2(tensor: &TensorProto) -> Array2<f32> {
    let data: Vec<f32> = tensor.float_data.clone(); // Extract data (adjust based on TensorProto type)
    
    // let shape = tensor.dims.clone(); // Get dimensions
    // let rows = shape[0] as usize;
    // let cols = shape[1] as usize;

    let shape = (
        tensor.dims[0] as usize,
        tensor.dims[1] as usize,
    );

    Array2::from_shape_vec(shape, data).unwrap() // Convert to ndarray
}


impl OnnxLoadable<GaussianNBModel> for GaussianNBModel
{
    // load GaussianNB model from ONNX proto model
    fn load_from_onnx_proto(model: onnx::ModelProto) -> GaussianNBModel {
        // extract the minimum data we need to add to the GaussianNBModel struct
        let graph = model.graph.expect("NO graph in the model");

        let initializer_classes = graph.initializer.get(0).expect("could not retrieve classes from onnx initializer object"); // get list of classes
        let initializer_theta = graph.initializer.get(1).expect("could not retrieve theta from onnx initializer object"); // get theta, i.e. the mean
        let initializer_sigma = graph.initializer.get(2).expect("could not retrieve sigma from onnx initializer object"); // get sigma, i.e. standard deviation
        let initializer_jointi = graph.initializer.get(3).expect("could not retrieve jointi from onnx initializer object");  // get jointi, i.e. log of prior probabilities

        let classes_array = convert_tensorproto_to_ndarray(initializer_classes);
        let theta_array = convert_tensorproto_to_ndarray_2(initializer_theta);
        let sigma_array = convert_tensorproto_to_ndarray_2(initializer_sigma);
        let jointi_array = convert_tensorproto_to_ndarray(initializer_jointi);


        GaussianNBModel {
            classes: classes_array, // list of classes
            mean: theta_array, // each row is a class, each column is a feature
            variance: sigma_array, // each row is a class, each column is a feature
            log_prior_prob: jointi_array // what proportion of trained data falls into each class
        }
    }
}


impl NaiveBayesAlgorithm<f32> for GaussianNBModel
{
    // run inference on given input (i.e. the unseen data)
    fn predict(&self, input: Array2<f32>) -> Array1<f32> {
        // vector for results
        let mut result = vec![];

        // iterate over every piece of unseen data to classify
        for (i, row) in input.rows().into_iter().enumerate(){
            // Follow the scikit learn implementation: https://github.com/scikit-learn/scikit-learn/blob/160fe6719a1f44608159b0999dea0e52a83e0963/sklearn/naive_bayes.py#L90

            let mut joint_log_likelihood = Vec::new();

            for i in 0..self.classes.len() {
                // Follow the scikit learn implementation: https://github.com/scikit-learn/scikit-learn/blob/160fe6719a1f44608159b0999dea0e52a83e0963/sklearn/naive_bayes.py#L509

                // Compute log prior probability
                let jointi = self.log_prior_prob[i];

                // First term: -0.5 * sum(log(2π * variance))
                let n_ij = -0.5 * self.variance.row(i).mapv(|var| (2.0 * PI as f32 * var).ln()).sum();

                // Second term: -0.5 * sum(((X - theta)²) / variance), summed over axis=1
                let deviation = row.mapv(|x| x) - self.mean.row(i).broadcast(row.raw_dim()).unwrap().mapv(|x| x);
                let n_ij_adjusted = n_ij - 0.5 * (deviation.mapv(|x| x.powi(2)) / self.variance.row(i)).sum_axis(Axis(1));

                // Store result
                joint_log_likelihood.push(n_ij_adjusted + jointi);
            }

            // Convert Vec<Array1<f64>> to Array2 (stack row-wise)
            let joint_log_likelihood_converted = stack(Axis(0), &joint_log_likelihood.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();

            // Transpose to match NumPy behavior: (samples x classes)
            let joint_log_likelihood_t = joint_log_likelihood_converted.t().to_owned();

            // Find the index of the max value in each row 
            let max_indices: Vec<usize> = joint_log_likelihood_t
                .axis_iter(Axis(1)) // Iterate over rows
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) // Find max index
                        .map(|(idx, _)| idx as usize) // Extract index
                        .unwrap()
                })
                .collect();   
            
            // Map max indices to class labels
            let predicted_class: f32 = self.classes[max_indices[0]];

            result.push(predicted_class)
        }

        Array1::from_vec(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_nb() {
        // get the onnx format of model
        let model = utils::load_onnx_model_proto("tests/resources/gaussian_nb.onnx").expect("Model has not been loaded");

        // extract most important data from the onnx model into a rust struct
        let gaussian_nb = GaussianNBModel::load_from_onnx_proto(model);

        let input = Array2::from_shape_vec(
            (3, 1), vec![
                2.0, 2.8, 3.5, // should be class 0
            ]
        ).unwrap();

        let prediction: Array1<f32> = gaussian_nb.predict(
            input
        );

        assert_eq!(vec![0.0], prediction.to_vec());
    }
}