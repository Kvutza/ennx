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

fn record() -> Record {
    Record {
        id: u64::MAX,
        shape: vec![2, 4],
        scale: 0.001,
        checkpoint: None,
    }
}

#[test]
fn native_values() {
    let record = record();
    let document = json!({"record":record,"nested":[null, {"enabled":true}, [1, 2]],
        "shape":record.shape,"value":record.scale});
    assert_eq!(document["record"]["id"].as_u64(), Some(u64::MAX));
    assert_eq!(record, from_value(document["record"].clone()).unwrap());
    assert_eq!(
        pointer(&document, "/nested/1/enabled").unwrap().as_bool(),
        Some(true)
    );
    let text = pretty_string(&document).unwrap();
    let parsed: Value = from_str(&text).unwrap();
    assert_eq!(
        record,
        from_value::<Record>(parsed["record"].clone()).unwrap()
    );
    assert_eq!(to_string(&document).unwrap(), to_string(&parsed).unwrap());
}

#[test]
fn unknown_fields() {
    assert!(from_str::<Record>(r#"{"id":42,"shape":[2],"scale":1.0,"extra":1}"#).is_err());
}

#[test]
fn line_pointers() {
    let value = json!({"a/b":{"~key":"bytes\u{0}remain"}});
    assert_eq!(pointer(&value, "/a~1b/~0key").unwrap(), "bytes\0remain");
    let mut output = Vec::new();
    write_line(&mut output, &value).unwrap();
    write_line(&mut output, &[1.0f32, 2.0]).unwrap();
    let text = std::str::from_utf8(&output).unwrap();
    assert_eq!(text.lines().count(), 2);
    assert_eq!(
        from_str::<Value>(text.lines().next().unwrap()).unwrap(),
        value
    );
}
