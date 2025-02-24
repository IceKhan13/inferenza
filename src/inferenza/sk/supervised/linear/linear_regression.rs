use ndarray::{Array1, Array2};
use crate::onnx;
use crate::onnx::{GraphProto, NodeProto, AttributeProto};
use crate::inferenza::OnnxLoadable;
use crate::inferenza::sk::Regressor;
use crate::inferenza::utils;


/// LinearRegression scikit learn model
#[derive(Debug)]
struct LinearRegressionModel<T> {
    coefficients: Array2<T>,
    intercepts: Array1<T>, // bias term
}


impl OnnxLoadable<LinearRegressionModel<f32>> for LinearRegressionModel<f32> {
    fn load_from_onnx_proto(path: &str) -> LinearRegressionModel<f32> {
        let model: onnx::ModelProto = utils::load_onnx_model_proto(path).expect("Model has not been loaded.");

        // load graph of the model
        let graph: GraphProto = model.graph.expect("No graph for model.");

        // first node is the node with regression coefficients and bias terms
        let node: &NodeProto = graph.node.get(0).expect("No elements in graph.");

        // get coeffs
        let coef_attributes: &AttributeProto = node.attribute.get(0) // 0 index for coefs
            .expect("No coefficient attribute.");

        // get intercepts
        let intercepts_attributes: &onnx::AttributeProto = node.attribute.get(1) // 1 index for intercepts
            .expect("No intercepts attribute."); // bias terms

        let coefficients: Vec<f32> = coef_attributes.floats.clone();
        let intercepts: Vec<f32> = intercepts_attributes.floats.clone();

        let coefficients_array: Array2<f32> = Array2::from_shape_vec((
            intercepts.len(), coefficients.len() / intercepts.len()
        ), coefficients).unwrap();
        let intercepts_array: Array1<f32> = Array1::from_vec(intercepts);

        LinearRegressionModel {
            coefficients: coefficients_array,
            intercepts: intercepts_array
        }
    }
}


impl Regressor<f32> for LinearRegressionModel<f32> {
    fn predict(&self, input: Array2<f32>) -> Array2<f32> {
        // todo!("Check if ndim == 1 do not transpose.");
        // X * W + bias
        input.dot(&self.coefficients.t()) + &self.intercepts
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_linear_regression() {
        let linear_regression: LinearRegressionModel<f32> = LinearRegressionModel::load_from_onnx_proto(
            "tests/resources/linear_regression.onnx"
        );

        let input: Array2<f32> = Array2::from_shape_vec(
            (3, 3), vec![
                0, 0, 1, // [1, 1]
                1, 1, 3, // [2, 2]
                2, 2, 2, // [3, 3]
            ]
        ).unwrap().mapv(|x| x as f32);


        let prediction: Array2<f32> = linear_regression.predict(input);

        assert_eq!(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], prediction.flatten().to_vec());
    } 
}
