use std::collections::HashMap;
use std::fs;
use std::env;
use std::hash::Hash;
use std::path::Path;
use std::io::{self, Read};
use std::str::FromStr;
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct InitializerParameter {
    dims: Vec<String>,
    data_type: u32,
    float_data: Option<Vec<f32>>,
    name: String
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Graph {
    initializer: Vec<InitializerParameter>
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct KmeansOnnx {
    ir_version: String,
    producer_version: String,
    graph: Graph,
}

fn euclidean_distance(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn closest_vector_index(target: &Vec<f64>, vectors: &Vec<Vec<f64>>) -> Option<usize> {
    let mut closest_index = None;
    let mut min_distance = f64::INFINITY;

    for (index, vector) in vectors.iter().enumerate() {
        let distance = euclidean_distance(target, vector);
        if distance < min_distance {
            min_distance = distance;
            closest_index = Some(index);
        }
    }

    closest_index
}

trait ClusterPredictor {
    fn predict(&self, input: &Vec<Vec<f64>>) -> Vec<i32>;
}

struct KmeansModel {
    cluster_centers: Vec<f64>,
    input_shape: Vec<u32>
}

impl ClusterPredictor for KmeansModel {
    fn predict(&self, input: &Vec<Vec<f64>>) -> Vec<i32> {
        todo!("Implement");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read() {
        let current_dir = env::current_dir().expect("Failed to get current directory");
        let relative_path = Path::new("tests/resources/kmeans.json");
        let absolute_path = current_dir.join(relative_path);

        match fs::read_to_string(absolute_path) {
            Ok(file_content) => {
                let kmeans: Result<KmeansOnnx, serde_json::Error> = serde_json::from_str(&file_content);
                println!("{:?}", kmeans);

            },
            Err(error) => eprintln!("Error reading file: {}", error)
        }
    }
}
