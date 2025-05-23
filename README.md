# Inferenza

A Rust based Machine Learning inferrer for SciKit Learn.


## Purpose

Inference latency is a key challenge in deploying machine learning models, particularly for real-time or interactive applications. High latency can lead to delayed responses, degraded user experience, or even system failures in time-sensitive environments. Complex models, especially deep neural networks, often require significant computational resources to process input and generate predictions, which can slow down inference. This issue is compounded when running on resource-constrained devices like mobile phones or embedded systems. The goal of this project is to develop a Rust based inferrer to help mitigate the effects of latency on at least a small part of the typical ML pipeline. Our initial focus is narrow for our first development phase: we aim to improve the speed of inference for the `predict()` methods of the most commonly used SciKit Learn models. The basic premise of this package is to allow users to pre-train their models using SciKit learn and then leverage Inferenza for predictions on unseen data. 


## Install

TODO

## Usage

Brodaly speaking there are four steps to using Inferenza in your project:

1. Pretrain the SciKit Learn model of your choice
2. Convert your model into ONNX format and save to an `.onnx` file
3. In your Rust project, load your model from the `.onnx` file using Inferenza's `load_from_onnx_proto()` method
4. Call Inferenza's corresponding `predict()` method for your model type with unseen inputs to make predictions

### Example using KMeans:

Step 1 - pretrain your model

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y, centers = make_blobs(n_samples=50, n_features=3, centers=5, cluster_std=0.5, shuffle=True, random_state=42,  return_centers=True)
num_features = X.shape[1]
num_clusters = centers.shape[0]

km = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=500, tol=1e-04, random_state=42)
km.fit(X)

y_km = km.predict(X)

```

Step 2 - save your model to a `.onnx` file

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Output filename for the exported ONNX model
out_onnx = "kmeans.onnx"

# In this example, we are creating a float32 input of any batch size with 3 features
initial_type = [('float_input', FloatTensorType([None, num_features]))]

# Convert the trained scikit-learn KMeans model (km) to an ONNX model
onnx_model = convert_sklearn(km, initial_types=initial_type)

# Add custom metadata to the ONNX model for identification
meta =  onnx_model.metadata_props.add()
meta.key = "sklearn_model"
meta.value = "KMeans"

# Save the serialized ONNX model to disk
with open(out_onnx, "wb") as f:
    f.write( onnx_model.SerializeToString())

```

Step 3 - in your Rust project, load the model from the `.onnx` file

```rust
let kmeans = KmeansModel::load_from_onnx_proto("<path_to_file>/kmeans.onnx");
```

Step 4 - make predictions

```rust
// define your input as a 3x3 array (3 inputs with 3 features each)
let input = Array2::from_shape_vec(
    (3, 3), vec![
        3.9755416, -9.76483, 9.557824, // this should be predicted as class 0
        3.9755416, -9.76483, 9.557824, // this should be predicted as class 0
        -2.466838, 9.029932, 4.4263315 // this should be predicted as class 1
    ]
).unwrap();

// make predictions on our inputs with kmeans predict() function
let prediction = kmeans.predict(
    input
);

// print prediction result
println!("{:?}", prediction.to_vec());
// output: [0, 0, 1]
```
