use super::*;

impl BpannBackend {
    pub fn index_snapshot(&self) -> Option<&BpannIndex> {
        self.index.indices.first()
    }

    pub fn page_bytes(&self) -> Vec<u8> {
        self.index
            .indices
            .first()
            .map(BpannIndex::page_bytes)
            .unwrap_or_default()
    }

    pub fn row_slice(&self, index: usize) -> Result<&[f64], BpannError> {
        self.train_x.row_slice(index)
    }

    pub fn y_yvar(&self, index: usize) -> Result<(&[f64], Option<&[f64]>), BpannError> {
        let y = self.train_y.row_slice(index)?;
        let yvar = self
            .train_yvar
            .as_ref()
            .map(|store| store.row_slice(index))
            .transpose()?;
        Ok((y, yvar))
    }

    pub fn index_bytes(&self) -> usize {
        self.index.index_bytes()
    }

    pub fn reopen(work_dir: PathBuf) -> Result<Self, BpannError> {
        let text = fs::read_to_string(work_dir.join("metadata.json"))
            .map_err(|error| BpannError::InvalidParameter(error.to_string()))?;
        let num_dim = crate::observation::parse_number(&text, "num_dim")
            .ok_or_else(|| BpannError::InvalidParameter("missing num_dim".into()))?;
        let num_metrics = crate::observation::parse_number(&text, "num_metrics")
            .ok_or_else(|| BpannError::InvalidParameter("missing num_metrics".into()))?;
        Self::new(
            work_dir,
            Array2::zeros((0, num_dim)),
            Array2::zeros((0, num_metrics)),
            None,
            text.contains("\"scale_x\":true"),
            Array1::ones(num_dim),
        )
    }

    pub fn with_soft(mut self, threshold: usize) -> Self {
        self.soft_threshold = threshold;
        if self.hard_threshold < threshold {
            self.hard_threshold = threshold;
        }
        self
    }

    pub fn with_hard(mut self, threshold: usize) -> Self {
        self.hard_threshold = threshold.max(self.soft_threshold);
        self
    }

    pub fn soft_threshold(&self) -> usize {
        self.soft_threshold
    }

    pub fn hard_threshold(&self) -> usize {
        self.hard_threshold
    }

    pub fn set_thresholds(&mut self, soft: usize, hard: usize) {
        let soft = soft.max(1);
        self.soft_threshold = soft;
        self.hard_threshold = hard.max(soft);
    }
}
