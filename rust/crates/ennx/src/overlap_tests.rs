use super::*;

#[test]
fn exact_overlap() {
    let a = Profile::new(b"aab", 1);
    assert_eq!(a.similarity(&Profile::new(b"abbb", 1)), 4.0 / 7.0);
    assert_eq!(a.similarity(&Profile::new(b"aab", 1)), 1.0);
    assert!(a.similarity(&Profile::new(&vec![b'a'; 4096], 1)) < 0.001);
}

#[test]
fn whitespace_credit() {
    let a = Profile::new(b"\0a\xff", 4);
    assert_eq!(a.similarity(&Profile::new(b"\n \t", 4)), 0.0);
    assert_eq!(a.similarity(&Profile::new(b" \0 a \xff ", 4)), 0.75);
    assert!(a.similarity(&Profile::new(b"\0a\xfe", 4)) < 0.75);
}

#[test]
fn contrast_controls() {
    let target = b"def increment(value):\n    return value + 1\n";
    let decoys = vec![
        b"class Container:\n    pass\n".to_vec(),
        b"raise RuntimeError('invalid')".to_vec(),
    ];
    let contrast = Contrast::new(target, &decoys, 4).unwrap();
    let reference = contrast.reward(target);
    assert!(reference > contrast.reward(&vec![b' '; 16384]));
    assert!(reference > contrast.reward(&vec![b':'; 16384]));
    assert!(reference > contrast.reward(&decoys[0]));
    assert!(contrast.reward(&decoys[0]) <= 0.0);
    assert!(Contrast::new(target, &decoys, 5).is_err());
}

#[test]
fn finite_components() {
    let contrast = Contrast::new(b"return item + 7", &[b"raise Exception()".to_vec()], 4).unwrap();
    for output in [b"".as_slice(), b"r", b"return item + 7"] {
        assert!(contrast.reward(output).is_finite());
    }
}
