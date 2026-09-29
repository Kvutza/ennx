use super::*;
use deser::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
struct Record {
    id: u64,
    shape: Vec<usize>,
    scale: f32,
    checkpoint: Option<String>,
}

#[test]
fn value_roundtrip() {
    let original: Record = from_str("id = 42\nshape = [2, 4]\nscale = 0.001\n").unwrap();
    assert!(original.checkpoint.is_none());
    let text = to_string(&original).unwrap();
    assert_eq!(original, from_str::<Record>(&text).unwrap());
    let value = to_value(&original).unwrap();
    assert_eq!(original, from_value::<Record>(value).unwrap());
    assert!(from_str::<Record>("id = 42\nshape = [2]\nscale = 1.0\nextra = 1").is_err());
}
