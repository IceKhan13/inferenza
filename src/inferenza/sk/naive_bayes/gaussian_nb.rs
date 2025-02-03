use ndarray::{Array1, Array2};
use crate::onnx;
use crate::inferenza::sk::naive_bayes::NaiveBayesAlgorithm;
use crate::inferenza::OnnxLoadable;
use crate::inferenza::utils;
use std::f64::consts::PI;

// struct for GaussianNB Scikit learn model, i.e. the minimum data we need from the model to make a prediction on unseen data
#[derive(Debug)] // Q: what does this do?
struct GaussianNBModel {
    mean: Array2<f32>,
    std: Array2<f32>,
    prior_prob: Array1<f32>,
}


// How gaussian nb works in practice:
// 1. Training Phase:
// For each class, calculate the mean and standard deviation of each feature from the training data. 
// Store these calculated parameters for later use in prediction. 
// 2. Prediction Phase:
// For a new data point:
//      For each class, calculate the likelihood of observing the data point's feature values based on the Gaussian distribution for that class. 
        // i.e. the prior probability, i.e. proportion of test data that was classified as one or other feature
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
            mean: [[123, 456], [234, 567]], // each row is a class, each column is a feature
            std_deviation: [[123, 456], [234, 567]], // each row is a class, each column is a feature
            prior_pr: [0.7, 0.3] // what proportion of trained data falls into each class
        }
    }
}


impl<T> NaiveBayesAlgorithm<T> for GaussianNBModel<T>
where
    T: ndarray::LinalgScalar + Clone + std::fmt::Debug + num::Float, // what does this do again?
{
    // run inference on given input (i.e. the unseen data)
    fn predict(&self, input: Array2<T>) -> Array1<u32> {
        // vector for results
        let mut result: Vec<u32> = vec![];

        // iterate over every piece of unseen data to classify
        for (i, row) in input.rows().into_iter().enumerate(){
            // 1. for each class and each feature in new data point, calc gaussian pdf
            // 2. calc posterior probability for each class using the gaussian pdfs
            // 3. final classification is the index of the largest posterior probability

            let post_probs: Vec<f32> = vec![]

            for (class_idx, prob) in prior_prob {

                let mut class_pdf: Vec<f32> = vec![] // collect the pdf calcs for each feature for this class

                for (feature_idx, feature) in row {
                    // get the train data mean for this class and this feature
                    let mean = self.mean[class_idx][feature_idx]
                    // get the train data standard deviation for this class and this feature
                    let std = self.std[class_idx][feature_idx]

                    // calculate pdf for this feature
                    let coeff = 1.0 / ((2.0 * PI * std.powi(2)).sqrt());
                    let exponent = (-((feature - mean).powi(2)) / (2.0 * std.powi(2))).exp();
                    let pdf = coeff * exponent

                    class_pdf.push(pdf)
                }

                // calculate posterior probability for this class
                let post_prob = self.prior_prob[class_idx] * class_pdf.iter().product()
                post_probs.push(post_prob)

            }

            // the largest value in the posterior probabilities is the most likely class the new data point is
            classification = post_probs.iter().enumerate().max_by_key(|&(_, &val)| val)
            result.push(classification)
        }

        Array1::from_vec(result)
    }
}

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