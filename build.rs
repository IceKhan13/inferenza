fn main() {
    prost_build::compile_protos(&["onnx.proto"], &["."])
        .expect("Failed to compile ONNX protobuf");
}