mod bundle;
mod steps;

use anyhow::Result;
use bundle::Bundle;

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).expect("onnx dir");
    let feats_path = std::env::args().nth(2).expect("features .bin");

    let b = Bundle::open(&dir)?;
    let raw = std::fs::read(&feats_path)?;
    let feats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let t = feats.len() / b.meta.steps.feature_dim;
    eprintln!("features {} x {}", t, b.meta.steps.feature_dim);

    let mut m = steps::StepModel::load(&b)?;
    let mem = m.memory(&b, &feats, t)?;
    let preds = m.decode_video(&b, &mem, t)?;
    for p in &preds {
        println!("{p}");
    }
    Ok(())
}
