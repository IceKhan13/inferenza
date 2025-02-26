use ndarray::{Array1, Array2};
use crate::onnx;
use crate::inferenza::sk::supervised::naive_bayes::NaiveBayesAlgorithm;
use crate::inferenza::OnnxLoadable;
use crate::inferenza::utils;
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
    sigma_sum_log: Array1<f32>
}

fn convert_tensorproto_to_ndarray(tensor: &TensorProto) -> Array1<f32> {
    let data: Vec<f32> = tensor.float_data.clone(); // Extract data (adjust based on TensorProto type)

    let shape = (
        tensor.dims[0] as usize,
    );
    Array1::from_vec(data)
}

fn convert_tensorproto_to_ndarray_2(tensor: &TensorProto) -> Array2<f32> {
    let data: Vec<f32> = tensor.float_data.clone(); // Extract data (adjust based on TensorProto type)

    let shape = (
        tensor.dims[1] as usize,
        tensor.dims[2] as usize,
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
        let initializer_sigma_sum_log = graph.initializer.get(4).expect("could not retrieve sigma_sum_log from onnx initializer object");


        let classes: Vec<f32> = initializer_classes.int32_data.iter().map(|x| *x as f32).collect();
        let classes_array = Array1::from_vec(classes);

        // let classes_array = convert_tensorproto_to_ndarray(initializer_classes);
        let theta_array = convert_tensorproto_to_ndarray_2(initializer_theta);
        let sigma_array = convert_tensorproto_to_ndarray_2(initializer_sigma);
        let jointi_array = convert_tensorproto_to_ndarray(initializer_jointi);
        let sigma_sum_log = convert_tensorproto_to_ndarray(initializer_sigma_sum_log);


        GaussianNBModel {
            classes: classes_array, // list of classes
            mean: theta_array, // each row is a class, each column is a feature
            variance: sigma_array, // each row is a class, each column is a feature
            log_prior_prob: jointi_array, // what proportion of trained data falls into each class
            sigma_sum_log: sigma_sum_log // intermediate calculation required for prediction
        }
    }
}

impl NaiveBayesAlgorithm<f32> for GaussianNBModel
{
    // run inference on given input (i.e. the unseen data)
    fn predict(&self, input: Array2<f32>) -> Array1<f32> {

        let mut joint_log_likelihood: Vec<Array2<f32>> = Vec::new();

        for i in 0..self.classes.len() {
            // Follow the scikit learn implementation: https://github.com/scikit-learn/scikit-learn/blob/160fe6719a1f44608159b0999dea0e52a83e0963/sklearn/naive_bayes.py#L509

            // get log of prior probability for this class
            let jointi = self.log_prior_prob[i];
            println!("JOINT_I FOR CLass I: {:?}", jointi);

            // Extract the i-th row from theta_ and var_
            let theta_row = self.mean.row(i);
            let var_row = self.variance.row(i);

            // Compute ((X - theta_row) ** 2) / var_row) along each row
            let squared_diff = (&input - &theta_row).mapv(|x| x.powi(2)); // Element-wise square
            let normalized = &squared_diff / &var_row; // Element-wise division

            // Sum along axis=1
            let summed = normalized.sum_axis(Axis(1)).insert_axis(Axis(1));

            let n_ij = self.sigma_sum_log[i] - 0.5 * summed;

            // Store result
            joint_log_likelihood.push(n_ij + jointi);
        }

        // Stack collected results into a single Array2
        let joint_log_likelihood = stack(Axis(0), &joint_log_likelihood.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();

        // Transpose to match NumPy behavior: (samples x classes)
        let joint_log_likelihood_t = joint_log_likelihood.t().to_owned();

        // Find the index of the max value for each input row
        let max_indices: Vec<usize> = joint_log_likelihood_t.axis_iter(Axis(1)) // Iterate over rows
            .map(|row| row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) // Find max index
                .map(|(idx, _)| idx)
                .unwrap() // Guaranteed to have a max in non-empty row
            )
            .collect();  

        // Map max indices to class labels
        let predicted_classes: Vec<f32> = max_indices.iter().map(|&idx| self.classes[idx]).collect();

        Array1::from_vec(predicted_classes)
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
            (3,4), vec![
                5.7,3.8,1.7,0.3, // class 0
                6.1,2.8,4.7,1.2	, // class 1
                7.7,2.6,6.9,2.3	 // class 2
            ]
        ).unwrap();

        let prediction = gaussian_nb.predict(input);


        assert_eq!(vec![0.0, 1.0, 2.0], prediction.to_vec());
    }
}