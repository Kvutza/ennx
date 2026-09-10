use bpann::distance::l2_f32;

#[test]
fn l2_vectors() {
    let left = [1.0, -2.0, 4.0, 0.5];
    let right = [-1.0, 1.0, 2.0, -0.5];
    assert_eq!(l2_f32(&left, &right), 18.0);
}
