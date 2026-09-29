use super::HEAD_DIM;

/// Real-valued per-head certificate accumulator. The Metal path must use FP32
/// ledgers (or fail closed); fixed-point atomics are not range-safe here.
#[derive(Clone, Copy)]
struct HeadLedger {
    z_hat: f64,
    s_hat: [f64; HEAD_DIM as usize],
    delta_z: f64,
    delta_s: f64,
}

impl Default for HeadLedger {
    fn default() -> Self {
        Self {
            z_hat: 0.0,
            s_hat: [0.0; HEAD_DIM as usize],
            delta_z: 0.0,
            delta_s: 0.0,
        }
    }
}

impl HeadLedger {
    fn output_bound(self) -> Option<f64> {
        if !self.z_hat.is_finite()
            || self.z_hat <= 0.0
            || !self.delta_z.is_finite()
            || self.delta_z < 0.0
            || !self.delta_s.is_finite()
            || self.delta_s < 0.0
            || self.s_hat.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let lower_z = self.z_hat - self.delta_z;
        if lower_z <= 0.0 {
            return None;
        }
        let s_norm = self
            .s_hat
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let bound = self.delta_s / lower_z + s_norm * self.delta_z / (lower_z * self.z_hat);
        bound.is_finite().then_some(bound)
    }
}

#[derive(Clone, Copy)]
struct NodeCharge {
    z_hat: f64,
    s_hat_scale: f64,
    delta_z: f64,
    delta_s: f64,
}

fn drop_charge(
    count: u32,
    log_center: f64,
    eta: f64,
    value_l1: f64,
    scale: f64,
) -> Option<NodeCharge> {
    if count == 0
        || !log_center.is_finite()
        || !eta.is_finite()
        || eta < 0.0
        || !value_l1.is_finite()
        || value_l1 < 0.0
        || !scale.is_finite()
        || log_center + eta > scale
    {
        return None;
    }
    let upper = (log_center + eta - scale).exp();
    if upper == 0.0 {
        return None;
    }
    let charge = NodeCharge {
        z_hat: 0.0,
        s_hat_scale: 0.0,
        delta_z: f64::from(count) * upper,
        delta_s: value_l1 * upper,
    };
    [charge.delta_z, charge.delta_s]
        .iter()
        .all(|value| value.is_finite())
        .then_some(charge)
}

fn monopole_charge(
    count: u32,
    log_center: f64,
    eta: f64,
    value_l1: f64,
    scale: f64,
) -> Option<NodeCharge> {
    let mut charge = drop_charge(count, log_center, eta, value_l1, scale)?;
    let center = (log_center - scale).exp();
    if center == 0.0 {
        return None;
    }
    charge.z_hat = f64::from(count) * center;
    charge.s_hat_scale = center;
    charge.delta_z *= eta;
    charge.delta_s *= eta;
    Some(charge)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monopole_bound() {
        let charge = monopole_charge(2, 0.0, 0.1, 2.0, 0.1).unwrap();
        let mut ledger = HeadLedger {
            z_hat: charge.z_hat,
            delta_z: charge.delta_z,
            delta_s: charge.delta_s,
            ..HeadLedger::default()
        };
        ledger.s_hat[0] = charge.s_hat_scale;
        ledger.s_hat[1] = charge.s_hat_scale;
        let bound = ledger.output_bound().unwrap();

        let exact_z = ((-0.1_f64).exp() + 0.1_f64.exp()) * (-0.1_f64).exp();
        let exact_s = [(-0.1_f64).exp(), 1.0].map(|value| value * (-0.1_f64).exp());
        let exact_output = exact_s.map(|value| value / exact_z);
        let approx_output = [0.5_f64, 0.5];
        let actual = exact_output
            .iter()
            .zip(approx_output)
            .map(|(&left, right)| (left - right) * (left - right))
            .sum::<f64>()
            .sqrt();
        assert!(actual <= bound, "actual {actual} exceeds bound {bound}");
    }

    #[test]
    fn fail_closed() {
        let dropped = drop_charge(64, 20.0, 2.0, 100.0, 22.0).unwrap();
        let ledger = HeadLedger {
            delta_z: dropped.delta_z,
            delta_s: dropped.delta_s,
            ..HeadLedger::default()
        };
        assert!(ledger.output_bound().is_none());
        assert!(drop_charge(64, 20.0, 2.0, 100.0, 21.9).is_none());
    }
}
