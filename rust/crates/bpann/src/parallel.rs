pub fn should_use_rayon() -> bool {
    rayon::current_num_threads() > 1
}

#[cfg(test)]
mod tests {
    use super::should_use_rayon;

    #[test]
    fn follows_current_pool_size() {
        let single_threaded = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-threaded pool");
        let multi_threaded = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("multi-threaded pool");

        assert!(!single_threaded.install(should_use_rayon));
        assert!(multi_threaded.install(should_use_rayon));
    }
}
