use ennx::prelude::{
    ENNParams, EpistemicNearestNeighbors, IndexDriver, PosteriorComputation, PosteriorFlags,
};
use ndarray::array;

fn main() -> Result<(), String> {
    let train_x = array![[0.0], [0.5], [1.0]];
    let train_y = array![[0.0], [1.0], [0.0]];
    let query = array![[0.25], [0.75]];
    let model = EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
        .map_err(|error| error.to_string())?;
    let params = ENNParams::new(2, 0.7, 0.13).map_err(|error| error.to_string())?;
    let flags = PosteriorFlags::new();
    let posterior = model
        .posterior(&query.view(), &params, &flags)
        .map_err(|error| error.to_string())?;
    println!("{:?}", posterior.mu);
    Ok(())
}
