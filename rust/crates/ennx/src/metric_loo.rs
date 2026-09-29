//! Ported from yubo-research/enn, commit 506e98c506eeb849cffbf53d9ddf3a3a799c6830.
//! Leave-one-out ENN log-likelihood for a diagonal metric.

const EPS: f64 = 1e-9;
const MIN_VAR: f64 = 1e-24;
const SCALE_GRID: [f64; 13] = [
    0.01,
    0.031_622_776_601_683_79,
    0.1,
    0.316_227_766_016_837_94,
    1.0,
    3.162_277_660_168_379_5,
    10.0,
    31.622_776_601_683_793,
    100.0,
    316.227_766_016_837_96,
    1000.0,
    3_162.277_660_168_379_5,
    10_000.0,
];
const NOISE_GRID: [f64; 7] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0];

fn variance(col: &[f64]) -> f64 {
    let n = col.len() as f64;
    let mean = col.iter().sum::<f64>() / n;
    let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    if var > MIN_VAR { var } else { 1.0 }
}

fn neighbors(z: &[f64], a_scaled: &[f64], n: usize, d: usize, k: usize) -> (Vec<usize>, Vec<f64>) {
    let mut nbr = vec![0usize; n * k];
    let mut d2n = vec![0.0; n * k];
    let mut dist = Vec::with_capacity(n - 1);
    for i in 0..n {
        dist.clear();
        for r in 0..n {
            if r == i {
                continue;
            }
            let distance = (0..d)
                .map(|j| a_scaled[j] * (z[i * d + j] - z[r * d + j]).powi(2))
                .sum();
            dist.push((distance, r));
        }
        if k > 0 {
            let order =
                |p: &(f64, usize), q: &(f64, usize)| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1));
            dist.select_nth_unstable_by(k - 1, order);
            dist[..k].sort_by(order);
            for t in 0..k {
                nbr[i * k + t] = dist[t].1;
                d2n[i * k + t] = dist[t].0;
            }
        }
    }
    (nbr, d2n)
}

fn column_loglik(ycol: &[f64], nbr: &[usize], d2n: &[f64], n: usize, k: usize) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for &scale in &SCALE_GRID {
        for &noise in &NOISE_GRID {
            let mut sum = 0.0;
            for i in 0..n {
                let mut wsum = 0.0;
                let mut wysum = 0.0;
                for t in 0..k {
                    let w = 1.0 / (EPS + (d2n[i * k + t] * scale).max(0.0) + noise);
                    wsum += w;
                    wysum += w * ycol[nbr[i * k + t]];
                }
                let mu = wysum / wsum;
                let var = 1.0 / wsum + noise;
                let err = ycol[i] - mu;
                sum += -0.5 * (2.0 * std::f64::consts::PI * var).ln() - 0.5 * err * err / var;
            }
            best = best.max(sum / n as f64);
        }
    }
    best
}

pub fn loo_loglik(x: &[f64], n: usize, d: usize, y: &[f64], m: usize, a: &[f64], k: usize) -> f64 {
    if n < 2
        || k == 0
        || a.len() != d
        || a.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || crate::metric_weights::validate_rows(x, n, d, y, m).is_err()
    {
        return f64::NEG_INFINITY;
    }
    let kk = k.min(n - 1);
    let mut mean = vec![0.0; d];
    let mut spr = vec![1.0; d];
    for j in 0..d {
        let col: Vec<f64> = (0..n).map(|i| x[i * d + j]).collect();
        mean[j] = col.iter().sum::<f64>() / n as f64;
        spr[j] = variance(&col);
    }
    let mut z = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            z[i * d + j] = (x[i * d + j] - mean[j]) / spr[j].sqrt();
        }
    }
    let a_scaled: Vec<f64> = (0..d).map(|j| a[j] * spr[j]).collect();
    let (nbr, d2n) = neighbors(&z, &a_scaled, n, d, kk);
    let mut total = 0.0;
    for col in 0..m {
        let raw: Vec<f64> = (0..n).map(|i| y[i * m + col]).collect();
        let yvar = variance(&raw);
        let ymean = raw.iter().sum::<f64>() / n as f64;
        let yz: Vec<f64> = raw.iter().map(|v| (v - ymean) / yvar.sqrt()).collect();
        total += column_loglik(&yz, &nbr, &d2n, n, kk);
    }
    total / m as f64
}

#[cfg(test)]
mod tests {
    #[test]
    fn partial_order() {
        let mut dist: Vec<(f64, usize)> = (0..50).map(|i| ((i * 17 % 50) as f64, i)).collect();
        dist.push((3.0, 3));
        let k = 10;
        let order = |p: &(f64, usize), q: &(f64, usize)| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1));
        let mut full = dist.clone();
        full.sort_by(order);
        let mut part = dist;
        part.select_nth_unstable_by(k - 1, order);
        part[..k].sort_by(order);
        assert_eq!(&full[..k], &part[..k]);
    }
}
