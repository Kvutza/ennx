//! Tracy profiling support for ENNX.
//!
//! ENNX starts the Tracy client automatically when an instrumented library
//! operation first runs. The client protocol matches Tracy 0.13.1.

use std::sync::OnceLock;

use tracy_client::{Client, PlotConfiguration, PlotLineStyle, Span};

pub use tracy_client::{frame_name, plot_name, span_location};

static CLIENT: OnceLock<Client> = OnceLock::new();

/// Return the process-wide Tracy client, starting it on first use.
pub fn client() -> &'static Client {
    CLIENT.get_or_init(|| {
        let client = Client::start();
        client.message("ENNX / Tracy 0.13.1", 0);
        for name in [
            tracy_client::plot_name!("ennx.optimizer.observations"),
            tracy_client::plot_name!("ennx.optimizer.candidates"),
            tracy_client::plot_name!("ennx.optimizer.arms"),
            tracy_client::plot_name!("ennx.knn.gpu_ns"),
            tracy_client::plot_name!("ennx.knn.scan_ns"),
            tracy_client::plot_name!("ennx.knn.select_ns"),
            tracy_client::plot_name!("ennx.knn.reduce_ns"),
            tracy_client::plot_name!("ennx.knn.rows"),
            tracy_client::plot_name!("ennx.knn.queries"),
            tracy_client::plot_name!("ennx.knn.dims"),
            tracy_client::plot_name!("ennx.knn.k"),
        ] {
            client.plot_config(
                name,
                PlotConfiguration::default().line_style(PlotLineStyle::Stepped),
            );
        }
        client
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn knn(profile: &crate::knn::KnnProfile) {
    let client = client();
    for (name, value) in [
        (
            tracy_client::plot_name!("ennx.knn.gpu_ns"),
            profile.gpu.as_nanos() as f64,
        ),
        (
            tracy_client::plot_name!("ennx.knn.scan_ns"),
            profile.scan.as_nanos() as f64,
        ),
        (
            tracy_client::plot_name!("ennx.knn.select_ns"),
            profile.select.as_nanos() as f64,
        ),
        (
            tracy_client::plot_name!("ennx.knn.reduce_ns"),
            profile.reduce.as_nanos() as f64,
        ),
        (
            tracy_client::plot_name!("ennx.knn.rows"),
            profile.rows as f64,
        ),
        (
            tracy_client::plot_name!("ennx.knn.queries"),
            profile.queries as f64,
        ),
        (
            tracy_client::plot_name!("ennx.knn.dims"),
            profile.dims as f64,
        ),
        (tracy_client::plot_name!("ennx.knn.k"), profile.k as f64),
    ] {
        client.plot(name, value);
    }
}

/// Enter a named CPU zone for the lifetime of the returned guard.
pub fn zone(location: &'static tracy_client::SpanLocation) -> Span {
    client().clone().span(location, 0)
}

pub(crate) fn stats(observations: usize, candidates: usize, arms: usize) {
    client().plot(
        tracy_client::plot_name!("ennx.optimizer.observations"),
        observations as f64,
    );
    client().plot(
        tracy_client::plot_name!("ennx.optimizer.candidates"),
        candidates as f64,
    );
    client().plot(tracy_client::plot_name!("ennx.optimizer.arms"), arms as f64);
}

#[cfg(test)]
mod tests {
    #[test]
    fn client() {
        assert!(std::ptr::eq(super::client(), super::client()));
    }

    #[test]
    fn zone() {
        let _zone = super::zone(tracy_client::span_location!("tracy.test"));
    }

    #[test]
    fn stats() {
        super::stats(1, 2, 3);
    }
}
