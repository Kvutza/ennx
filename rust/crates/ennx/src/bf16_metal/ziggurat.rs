use std::fmt::Write as _;
use std::sync::LazyLock;

pub(super) const ZIGGURAT_LAYERS: usize = 256;
pub(super) const ZIGGURAT_R: f64 = 3.654_152_885_361_009;
pub(super) const ZIGGURAT_V: f64 = 0.004_928_673_233_99;

pub(super) struct ZigguratTables {
    pub(super) thresholds: [u32; ZIGGURAT_LAYERS],
    pub(super) widths: [f32; ZIGGURAT_LAYERS],
    pub(super) densities: [f32; ZIGGURAT_LAYERS],
}

pub(super) static ZIGGURAT: LazyLock<ZigguratTables> = LazyLock::new(|| {
    let mut thresholds = [0; ZIGGURAT_LAYERS];
    let mut widths = [0.0; ZIGGURAT_LAYERS];
    let mut densities = [0.0; ZIGGURAT_LAYERS];
    let scale = 2_147_483_648.0;
    let mut boundary = ZIGGURAT_R;
    let mut previous = boundary;
    let tail_density = (-0.5 * boundary * boundary).exp();
    let tail_width = ZIGGURAT_V / tail_density;

    thresholds[0] = (boundary / tail_width * scale) as u32;
    thresholds[1] = 0;
    widths[0] = (tail_width / scale) as f32;
    widths[ZIGGURAT_LAYERS - 1] = (boundary / scale) as f32;
    densities[0] = 1.0;
    densities[ZIGGURAT_LAYERS - 1] = tail_density as f32;

    for layer in (1..ZIGGURAT_LAYERS - 1).rev() {
        boundary =
            (-2.0 * (ZIGGURAT_V / boundary + (-0.5 * boundary * boundary).exp()).ln()).sqrt();
        thresholds[layer + 1] = (boundary / previous * scale) as u32;
        previous = boundary;
        densities[layer] = (-0.5 * boundary * boundary).exp() as f32;
        widths[layer] = (boundary / scale) as f32;
    }
    ZigguratTables {
        thresholds,
        widths,
        densities,
    }
});

pub(super) fn metal_tables() -> String {
    fn uint_array(name: &str, values: &[u32]) -> String {
        let mut source = format!("constant uint {name}[{}] = {{", values.len());
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                source.push(',');
            }
            write!(source, "{value}u").expect("writing a String cannot fail");
        }
        source.push_str("};\n");
        source
    }

    fn float_array(name: &str, values: &[f32]) -> String {
        let mut source = format!("constant float {name}[{}] = {{", values.len());
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                source.push(',');
            }
            write!(source, "{value:?}f").expect("writing a String cannot fail");
        }
        source.push_str("};\n");
        source
    }

    let mut source = uint_array("ziggurat_thresholds", &ZIGGURAT.thresholds);
    source.push_str(&float_array("ziggurat_widths", &ZIGGURAT.widths));
    source.push_str(&float_array("ziggurat_densities", &ZIGGURAT.densities));
    source
}
