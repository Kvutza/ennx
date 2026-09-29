//! TOML uses the same native Deser value tree as JSON.

pub use deser_toml::{from_str, to_string};
pub use deser_value::{Kind, Map as Table, Value};

pub fn from_value<T: for<'de> deser::Deserialize<'de>>(value: Value) -> Result<T, deser::Error> {
    deser_value::from_value(&value)
}

pub fn to_value<T: deser::Serialize>(value: &T) -> Result<Value, deser::Error> {
    deser_value::to_value(value)
}

pub fn pretty_string<T: deser::Serialize + ?Sized>(value: &T) -> Result<String, deser::Error> {
    deser_toml::to_string(&value)
}

#[cfg(test)]
#[path = "toml/tests.rs"]
mod tests;
