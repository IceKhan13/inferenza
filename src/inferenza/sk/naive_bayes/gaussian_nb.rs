use ndarray::{Array1, Array2};
use crate::onnx;
use crate::inferenza::sk::naive_bayes::NaiveBayesAlgorithm;
use crate::inferenza::OnnxLoadable;
use crate::inferenza::utils;

// struct for GaussianNB Scikit learn model, i.e. the minimum data we need from the model to make a prediction on unseen data
#[derive(Debug)] // Q: what does this do?
struct GaussianNBModel {
    test: u32,
}


// How gaussian nb works in practice:
// 1. Training Phase:
// For each class, calculate the mean and standard deviation of each feature from the training data. 
// Store these calculated parameters for later use in prediction. 
// 2. Prediction Phase:
// For a new data point:
//      For each class, calculate the likelihood of observing the data point's feature values based on the Gaussian distribution for that class. 
//      Multiply these likelihoods by the prior probability of each class. 
//      Select the class with the highest resulting probability as the prediction. 


impl OnnxLoadable<GaussianNBModel> for GaussianNBModel
{
    // load GaussianNB model from ONNX proto model
    fn load_from_onnx_proto(model: onnx::ModelProto) -> GaussianNBModel {
        // extract the minimum data we need to add to the GaussianNBModel struct
        println!("hello!");
        println!("{}", model.graph); //how do i look at what the model object is??

        GaussianNBModel {
            test: 123,
        }
    }
}


// impl<T> NaiveBayesAlgorithm<T> for GaussianNBModel<T>
// where
//     T: ndarray::LinalgScalar + Clone + std::fmt::Debug + num::Float, // what does this do again?
// {
//     // run inference on given input (i.e. the unseen data)
//     fn predict(&self, input: Array2<T>) -> Array1<u32> {
//         // vector for results
//         let mut result: Vec<u32> = vec![];

//         // iterate over every piece of unseen data to classify
//         // for (i, row) in input.rows().into_iter().enumerate(){
//         //     // TODO the logic for classifying a new datapoint
//         // }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans() {
        // get the onnx format of model
        let model = utils::load_onnx_model_proto("test/resources/gaussian_nb.onnx").expect("Model has not been loaded");

        // extract most important data from the onnx model into a rust struct
        let gaussian_nb = GaussianNBModel::load_from_onnx_proto(model);

        // let input = Array2::from_shape_vec(...)

        // let prediction = gaussian_nb.predict(input)

        // assert_eq!(...)
    }
}