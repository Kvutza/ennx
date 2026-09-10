use ndarray::array;
use std::sync::Mutex;
use tempfile::TempDir;

#[test]
fn check_rows() {
    ennx::backend::disk_observation::check_rows(10).unwrap();
}

#[test]
fn read_stale() {
    let stale = Mutex::new(false);
    assert!(!ennx::backend::disk_observation::read_stale(&stale));
    ennx::backend::disk_observation::set_stale(&stale);
    assert!(ennx::backend::disk_observation::read_stale(&stale));
}

#[test]
fn open_yvar() {
    let dir = TempDir::new().unwrap();
    let yv = array![[0.1]];
    ennx::backend::disk_observation::open_yvar(dir.path(), 1, Some(&yv)).unwrap();
}

#[test]
fn check_backend() {
    let dir = TempDir::new().unwrap();
    ennx::backend::disk_observation::write_metadata(dir.path(), 0, 4, 1, false, 0, "bpann_disk")
        .unwrap();
    ennx::backend::disk_observation::check_backend(dir.path(), "bpann_disk").unwrap();
}

#[test]
fn append_yvar() {
    let dir = TempDir::new().unwrap();
    let mut yvar = ennx::backend::disk_observation::open_yvar(dir.path(), 1, None).unwrap();
    ennx::backend::disk_observation::append_yvar(
        dir.path(),
        1,
        &mut yvar,
        Some(&array![[0.2]].view()),
    )
    .unwrap();
}

#[test]
fn mark_dirty() {
    let dirty = Mutex::new(false);
    ennx::backend::disk_observation::mark_dirty(&dirty);
    assert!(*dirty.lock().unwrap());
}

#[test]
fn load_rows() {
    let dir = TempDir::new().unwrap();
    ennx::backend::disk_observation::write_metadata(dir.path(), 0, 4, 1, false, 2, "bpann_disk")
        .unwrap();
    assert_eq!(
        ennx::backend::disk_observation::load_rows(dir.path()),
        Some(2)
    );
}

#[test]
fn load_backend() {
    let dir = TempDir::new().unwrap();
    ennx::backend::disk_observation::write_metadata(dir.path(), 0, 4, 1, false, 0, "bpann_disk")
        .unwrap();
    assert_eq!(
        ennx::backend::disk_observation::load_backend(dir.path()).as_deref(),
        Some("bpann_disk")
    );
}

#[test]
fn write_metadata() {
    let dir = TempDir::new().unwrap();
    ennx::backend::disk_observation::write_metadata(dir.path(), 0, 4, 1, false, 0, "bpann_disk")
        .unwrap();
}

#[test]
fn check_dims() {
    ennx::backend::disk_observation::check_dims(4, 1).unwrap();
}

#[test]
fn parse_number() {
    let text = r#"{"num_obs":42,"index_backend":"bpann_disk"}"#;
    assert_eq!(
        ennx::backend::disk_observation::parse_number(text, "num_obs"),
        Some(42)
    );
}

#[test]
fn parse_string() {
    let text = r#"{"index_backend":"bpann_disk"}"#;
    assert_eq!(
        ennx::backend::disk_observation::parse_string(text, "index_backend").as_deref(),
        Some("bpann_disk")
    );
}
