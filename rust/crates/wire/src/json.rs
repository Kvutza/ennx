//! JSON I/O and dynamic values share one native Deser data model.

pub use crate::wire_json as json;
use deser::{Deserialize, Error, Serialize};
pub use deser_json::{from_reader, from_slice, from_str, to_string, to_writer};
pub use deser_value::{Kind, Map, Value};

/// Construct native Deser values while borrowing serializable expressions.
/// Parsing and encoding remain entirely in Deser.
#[macro_export]
macro_rules! wire_json {
    (@map $map:ident;) => {};
    (@map $map:ident; $key:tt : null $(, $($rest:tt)*)?) => {
        $map.insert($key, $crate::json::Value::null());
        $crate::wire_json!(@map $map; $($($rest)*)?);
    };
    (@map $map:ident; $key:tt : {$($inner:tt)*} $(, $($rest:tt)*)?) => {
        $map.insert($key, $crate::wire_json!({$($inner)*}));
        $crate::wire_json!(@map $map; $($($rest)*)?);
    };
    (@map $map:ident; $key:tt : [$($inner:tt)*] $(, $($rest:tt)*)?) => {
        $map.insert($key, $crate::wire_json!([$($inner)*]));
        $crate::wire_json!(@map $map; $($($rest)*)?);
    };
    (@map $map:ident; $key:tt : $value:expr $(, $($rest:tt)*)?) => {
        $map.insert($key, $crate::wire_json!($value));
        $crate::wire_json!(@map $map; $($($rest)*)?);
    };
    (@seq $seq:ident;) => {};
    (@seq $seq:ident; null $(, $($rest:tt)*)?) => {
        $seq.push($crate::json::Value::null());
        $crate::wire_json!(@seq $seq; $($($rest)*)?);
    };
    (@seq $seq:ident; {$($inner:tt)*} $(, $($rest:tt)*)?) => {
        $seq.push($crate::wire_json!({$($inner)*}));
        $crate::wire_json!(@seq $seq; $($($rest)*)?);
    };
    (@seq $seq:ident; [$($inner:tt)*] $(, $($rest:tt)*)?) => {
        $seq.push($crate::wire_json!([$($inner)*]));
        $crate::wire_json!(@seq $seq; $($($rest)*)?);
    };
    (@seq $seq:ident; $value:expr $(, $($rest:tt)*)?) => {
        $seq.push($crate::wire_json!($value));
        $crate::wire_json!(@seq $seq; $($($rest)*)?);
    };
    (null) => { $crate::json::Value::null() };
    ({$($items:tt)*}) => {{
        let mut object = $crate::json::Map::new();
        $crate::wire_json!(@map object; $($items)*);
        $crate::json::Value::from(object)
    }};
    ([$($items:tt)*]) => {{
        let mut array: Vec<$crate::json::Value> = Vec::new();
        $crate::wire_json!(@seq array; $($items)*);
        $crate::json::Value::from(array)
    }};
    ($value:expr) => {
        $crate::json::to_value(&$value).expect("JSON value must be serializable")
    };
}

pub fn write_line<W: std::io::Write, T: Serialize + ?Sized>(
    mut writer: W,
    value: &T,
) -> Result<(), Error> {
    to_writer(&mut writer, &value)?;
    writer.write_all(b"\n").map_err(Error::from)
}

pub fn from_value<T: for<'de> Deserialize<'de>>(value: Value) -> Result<T, Error> {
    deser_value::from_value(&value)
}

pub fn to_value<T: Serialize>(value: T) -> Result<Value, Error> {
    deser_value::to_value(&value)
}

pub fn to_vec<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, Error> {
    to_string(&value).map(String::into_bytes)
}

pub fn pretty_vec<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, Error> {
    pretty_string(value).map(String::into_bytes)
}

pub fn pretty_string<T: Serialize + ?Sized>(value: &T) -> Result<String, Error> {
    deser_json::SerializerConfig::new()
        .pretty(deser_json::Indent::Spaces(2))
        .to_string(&value)
}

pub fn pretty_writer<W: std::io::Write, T: Serialize + ?Sized>(
    writer: W,
    value: &T,
) -> Result<(), Error> {
    deser_json::SerializerConfig::new()
        .pretty(deser_json::Indent::Spaces(2))
        .to_writer(writer, &value)
}

pub fn pointer<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(value);
    }
    let mut current = value;
    for segment in path.strip_prefix('/')?.split('/') {
        let key = segment.replace("~1", "/").replace("~0", "~");
        current = if current.is_seq() {
            current.get(key.parse::<usize>().ok()?)?
        } else {
            current.get(key.as_str())?
        };
    }
    Some(current)
}

#[cfg(test)]
#[path = "json/tests.rs"]
mod tests;
