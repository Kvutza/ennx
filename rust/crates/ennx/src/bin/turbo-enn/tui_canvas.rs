//! High-resolution terminal canvas inspired by rsille and Ratatui.
//! Uses Unicode Braille patterns (U+2800..U+28FF) for 8x sub-pixel resolution (2x4 dots/cell)
//! and block-element gauges for rich terminal telemetry.

pub(super) struct BrailleCanvas {
    pub width: usize,
    pub height: usize,
    dots: Vec<u8>,
}

impl BrailleCanvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            dots: vec![0; width.saturating_mul(height)],
        }
    }

    pub fn set(&mut self, x: usize, y: usize) {
        let (cx, cy) = (x / 2, y / 4);
        if cx < self.width && cy < self.height {
            let mask = match (x % 2, y % 4) {
                (0, 0) => 0x01,
                (0, 1) => 0x02,
                (0, 2) => 0x04,
                (0, 3) => 0x40,
                (1, 0) => 0x08,
                (1, 1) => 0x10,
                (1, 2) => 0x20,
                (1, 3) => 0x80,
                _ => 0,
            };
            let idx = cy * self.width + cx;
            if idx < self.dots.len() {
                self.dots[idx] |= mask;
            }
        }
    }

    pub fn sparkline(&mut self, values: &[f64]) {
        if values.is_empty() || self.width == 0 || self.height == 0 {
            return;
        }
        let total_x = self.width * 2;
        let total_y = self.height * 4;
        let min_v = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let span = (max_v - min_v).max(1e-6);
        let step = values.len() as f64 / total_x as f64;
        for dx in 0..total_x {
            let idx = ((dx as f64 * step) as usize).min(values.len() - 1);
            let norm = ((values[idx] - min_v) / span).clamp(0.0, 1.0);
            let dy = ((1.0 - norm) * (total_y - 1) as f64).round() as usize;
            self.set(dx, dy.min(total_y - 1));
        }
    }

    pub fn bar(&mut self, fraction: f64) {
        let total_dots = (fraction.clamp(0.0, 1.0) * (self.width * 2) as f64).round() as usize;
        let total_y = self.height * 4;
        for dx in 0..total_dots {
            for dy in 0..total_y {
                self.set(dx, dy);
            }
        }
    }

    pub fn render(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.height);
        for row in 0..self.height {
            let mut line = String::with_capacity(self.width * 4);
            for col in 0..self.width {
                let mask = self.dots.get(row * self.width + col).copied().unwrap_or(0);
                line.push(char::from_u32(0x2800 + u32::from(mask)).unwrap_or(' '));
            }
            lines.push(line);
        }
        lines
    }
}

pub(super) fn inline_sparkline(values: &[f64], width: usize) -> String {
    if values.is_empty() || width == 0 {
        return " ".repeat(width);
    }
    let mut canvas = BrailleCanvas::new(width, 1);
    canvas.sparkline(values);
    canvas.render().into_iter().next().unwrap_or_default()
}

pub(super) fn progress_gauge(fraction: f64, width: usize) -> String {
    let clamped = fraction.clamp(0.0, 1.0);
    let filled_chars = (clamped * width as f64).round() as usize;
    let full = "█".repeat(filled_chars.min(width));
    let empty = "░".repeat(width.saturating_sub(filled_chars));
    format!("{full}{empty}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip() {
        let mut canvas = BrailleCanvas::new(10, 5);
        canvas.set(usize::MAX, usize::MAX);
        canvas.set(1000, 1000);
        canvas.set(0, 0);
        canvas.set(19, 19);
        assert_eq!(canvas.render().len(), 5);
    }

    #[test]
    fn test_unicode() {
        let mut canvas = BrailleCanvas::new(8, 2);
        canvas.sparkline(&[0.1, 0.5, 0.2, 0.9, 0.4]);
        for line in canvas.render() {
            for ch in line.chars() {
                let code = ch as u32;
                assert!((0x2800..=0x28FF).contains(&code));
            }
        }
    }

    #[test]
    fn test_monotonic() {
        let mut canvas = BrailleCanvas::new(4, 2);
        canvas.set(1, 1);
        let before = canvas.dots.iter().copied().sum::<u8>();
        canvas.set(2, 2);
        let after = canvas.dots.iter().copied().sum::<u8>();
        assert!(after > before);
    }

    #[test]
    fn test_sparkline() {
        let spark = inline_sparkline(&[1.0, 2.0, 3.0, 4.0], 4);
        assert_eq!(spark.chars().count(), 4);
        for ch in spark.chars() {
            let code = ch as u32;
            assert!((0x2800..=0x28FF).contains(&code));
        }
    }

    #[test]
    fn test_bounds() {
        assert_eq!(progress_gauge(-0.5, 10).chars().count(), 10);
        assert_eq!(progress_gauge(1.5, 10).chars().count(), 10);
        assert_eq!(progress_gauge(0.5, 10), "█████░░░░░");
    }
}
