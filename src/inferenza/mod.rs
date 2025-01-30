mod sk;
pub mod utils;

use crate::onnx::ModelProto;

/// Trait for ONNX loadable structs
trait OnnxLoadable<K> {
    /// Loads struct from ONNX model proto
    fn load_from_onnx_proto(model: ModelProto) -> K;
}
