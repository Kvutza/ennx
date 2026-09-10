pub fn use_rayon() -> bool {
    rayon::current_num_threads() > 1
}

#[cfg(test)]
mod tests {
    use super::use_rayon;

    #[test]
    fn follows_size() {
        let single_threaded = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-threaded pool");
        let multi_threaded = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("multi-threaded pool");

        assert!(!single_threaded.install(use_rayon));
        assert!(multi_threaded.install(use_rayon));
    }
}
