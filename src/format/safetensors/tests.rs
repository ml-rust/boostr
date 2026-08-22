use super::*;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;
use std::io::Write;
use tempfile::NamedTempFile;

fn create_test_file() -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();

    let header = serde_json::json!({
        "__metadata__": { "format": "pt" },
        "weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
        }
    });
    let header_str = header.to_string();
    let header_bytes = header_str.as_bytes();

    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header_bytes).unwrap();

    for f in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        file.write_all(&f.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
    file
}

#[test]
fn test_open_and_metadata() {
    let f = create_test_file();
    let st = SafeTensors::open(f.path()).unwrap();
    assert_eq!(st.len(), 1);
    assert_eq!(st.metadata().get("format"), Some(&"pt".to_string()));
}

#[test]
fn test_tensor_info() {
    let f = create_test_file();
    let st = SafeTensors::open(f.path()).unwrap();
    let info = st.tensor_info("weight").unwrap();
    assert_eq!(info.dtype, DType::F32);
    assert_eq!(info.shape, vec![2, 3]);
    assert_eq!(info.numel(), 6);
    assert_eq!(info.size_bytes(), 24);
}

#[test]
fn test_load_tensor_f32() {
    let (_, device) = cpu_setup();
    let f = create_test_file();
    let mut st = SafeTensors::open(f.path()).unwrap();
    let tensor = st.load_tensor::<CpuRuntime>("weight", &device).unwrap();
    assert_eq!(tensor.shape(), &[2, 3]);
    let data = tensor.to_vec::<f32>();
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[5] - 6.0).abs() < 1e-6);
}

fn create_test_file_bf16() -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();

    let header = serde_json::json!({
        "__metadata__": { "format": "pt" },
        "weight": {
            "dtype": "BF16",
            "shape": [2, 3],
            "data_offsets": [0, 12]
        }
    });
    let header_str = header.to_string();
    let header_bytes = header_str.as_bytes();

    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header_bytes).unwrap();

    for f in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        file.write_all(&half::bf16::from_f32(f).to_le_bytes())
            .unwrap();
    }
    file.flush().unwrap();
    file
}

#[test]
fn test_load_tensor_bf16() {
    let (_, device) = cpu_setup();
    let f = create_test_file_bf16();
    let mut st = SafeTensors::open(f.path()).unwrap();
    let tensor = st.load_tensor::<CpuRuntime>("weight", &device).unwrap();
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.dtype(), DType::BF16);
    let data: Vec<half::bf16> = tensor.to_vec();
    assert!((data[0].to_f32() - 1.0).abs() < 1e-2);
    assert!((data[5].to_f32() - 6.0).abs() < 1e-2);
}

#[test]
fn test_tensor_not_found() {
    let f = create_test_file();
    let st = SafeTensors::open(f.path()).unwrap();
    assert!(st.tensor_info("nonexistent").is_err());
}

#[test]
fn test_save_and_load_roundtrip() {
    let (_, device) = cpu_setup();
    let tmp = NamedTempFile::new().unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "w1".to_string(),
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device),
    );

    save_safetensors(tmp.path(), &tensors, None).unwrap();

    let mut loaded = SafeTensors::open(tmp.path()).unwrap();
    assert_eq!(loaded.len(), 1);
    let t = loaded.load_tensor::<CpuRuntime>("w1", &device).unwrap();
    assert_eq!(t.shape(), &[2, 2]);
    let data = t.to_vec::<f32>();
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[3] - 4.0).abs() < 1e-6);
}
