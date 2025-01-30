use crate::onnx;
use std::fs::File;
use std::io::Read;
use prost::Message;
use std::any::TypeId;


/// Loads ONNX proto model from file
pub fn load_onnx_model_proto(path: &str) -> Result<onnx::ModelProto, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let model = onnx::ModelProto::decode(&*buffer)?;
    
    Ok(model)
}


#[cfg(test)]
mod tests {
    use std::any::Any;

    use onnx::ModelProto;

    use super::*;

    #[test]
    fn test_kmeans() {
        let model = load_onnx_model_proto("tests/resources/kmeans.onnx")
            .expect("Model has not been loaded.");

        assert_eq!(model.type_id(), TypeId::of::<ModelProto>());
    }
}