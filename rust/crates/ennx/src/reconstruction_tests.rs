use super::*;

fn oracle<T: Eq>(pattern: &[T], tokens: &[T]) -> usize {
    let mut row = (0..=pattern.len()).collect::<Vec<_>>();
    for (position, token) in tokens.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = position + 1;
        for (column, expected) in pattern.iter().enumerate() {
            let previous = row[column + 1];
            row[column + 1] = (diagonal + usize::from(token != expected))
                .min(previous + 1)
                .min(row[column] + 1);
            diagonal = previous;
        }
    }
    row[pattern.len()]
}

#[test]
fn word_boundaries() {
    use rand::{Rng, SeedableRng};
    let mut random = rand::rngs::StdRng::from_entropy();
    for length in [0usize, 1, 31, 63, 64, 65, 127, 128, 129, 255, 4096] {
        let pattern = (0..length)
            .map(|_| random.gen_range(0..16))
            .collect::<Vec<_>>();
        let reference = Reference::new(&pattern);
        for output in [length.saturating_sub(7), length, length + 3] {
            let tokens = (0..output)
                .map(|_| random.gen_range(0..16))
                .collect::<Vec<_>>();
            assert_eq!(reference.distance(&tokens), oracle(&pattern, &tokens));
        }
        assert_eq!(reference.reward(&pattern), (0, 1.0));
    }
}

#[test]
fn edit_alignment() {
    let expected = (0..4096).collect::<Vec<_>>();
    let reference = Reference::new(&expected);
    let mut shifted = expected.clone();
    shifted.insert(63, 9000);
    assert_eq!(reference.distance(&shifted), 1);
    shifted.remove(129);
    assert_eq!(reference.distance(&shifted), 2);
    assert_eq!(reference.distance(&[]), 4096);
    assert_eq!(Reference::new(&[]).distance(&shifted), 4096);
}

#[test]
fn byte_distance() {
    let reference = Reference::new(b"return value\xff");
    assert_eq!(reference.reward(b"return value\xff"), (0, 1.0));
    assert_eq!(reference.distance(b"return value\xfe"), 1);
    assert_eq!(reference.distance(b"return value\xef\xbf\xbd"), 3);
    assert!(reference.reward(b"return other\xff").1 > reference.reward(b"xxxxx other\xff").1);
}

#[test]
fn byte_boundaries() {
    use rand::{Rng, SeedableRng};
    let mut random = rand::rngs::StdRng::from_entropy();
    for length in [0usize, 63, 64, 65, 127, 128, 129, 255, 1024] {
        let pattern = (0..length)
            .map(|_| random.r#gen::<u8>())
            .collect::<Vec<_>>();
        let tokens = (0..length + 3)
            .map(|_| random.r#gen::<u8>())
            .collect::<Vec<_>>();
        assert_eq!(
            Reference::new(&pattern).distance(&tokens),
            oracle(&pattern, &tokens)
        );
    }
}

#[test]
fn repetition_control() {
    let expected = (0..4096).map(|i| i % 256).collect::<Vec<_>>();
    let reference = Reference::new(&expected);
    let repeated = vec![0; 4096];
    assert_eq!(reference.distance(&repeated), 4080);
    assert!(reference.reward(&repeated).1 < 0.01);
    let unrelated = vec![8191; 4096];
    assert_eq!(reference.reward(&unrelated), (4096, 0.0));
    assert_eq!(reference.reward(&expected), (0, 1.0));
}

#[test]
#[ignore = "explicit CPU reward timing, not a full BO latency result"]
fn length_timing() {
    use rand::{Rng, SeedableRng};
    let mut random = rand::rngs::StdRng::from_entropy();
    let pattern = (0..4096)
        .map(|_| random.gen_range(0..8192))
        .collect::<Vec<_>>();
    let tokens = (0..4096)
        .map(|_| random.gen_range(0..8192))
        .collect::<Vec<_>>();
    let start = std::time::Instant::now();
    let reference = Reference::new(&pattern);
    let preparation_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut times = Vec::new();
    for _ in 0..11 {
        let start = std::time::Instant::now();
        std::hint::black_box(reference.reward(std::hint::black_box(&tokens)));
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    eprintln!(
        "ENNX_RECONSTRUCTION_PROFILE preparation_ms={preparation_ms:.3} median_ms={:.3} max_ms={:.3} tokens=4096 words=64",
        times[5], times[10]
    );
}
