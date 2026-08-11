//! The exported ONNX bundle: graphs, constant tables and the tags that decide
//! how they are driven.
//!
//! Everything here is produced by `uv run pitvis-export`. The tables are raw
//! little-endian f32 with their lengths in pipeline.json, precisely so this
//! side needs no .npy parser and no array crate — see `_bins` in export.py.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct Steps {
    pub space: String,
    pub feature_dim: usize,
    pub width: usize,
    pub mask_excluded: bool,
    pub excluded: Vec<i64>,
    pub cci_n: usize,
    pub sos: usize,
    pub num_classes: usize,
    pub d_model: usize,
}

#[derive(Debug, Deserialize)]
pub struct Instruments {
    pub feature_dim: usize,
    pub window: usize,
    pub num_instruments: usize,
    pub per_class_thresholds: bool,
    pub threshold: f32,
}

#[derive(Debug, Deserialize)]
pub struct Transform {
    pub input_size: (usize, usize, usize),
    pub mean: (f32, f32, f32),
    pub std: (f32, f32, f32),
    pub interpolation: String,
}

#[derive(Debug, Deserialize)]
pub struct Backbone {
    pub space: String,
    pub backbone: String,
    pub feature_dim: usize,
    pub transform: Transform,
}

#[derive(Debug, Deserialize)]
pub struct Pipeline {
    pub steps: Steps,
    pub instruments: Instruments,
    pub backbone: Backbone,
}

pub struct Tables {
    pub pe: Vec<f32>,          // (max_len, d_model), row-major
    pub phase: Vec<f32>,       // (num_classes + 1, d_model)
    pub steps_mean: Vec<f32>,
    pub steps_std: Vec<f32>,
    pub inst_mean: Vec<f32>,
    pub inst_std: Vec<f32>,
    pub inst_tau: Vec<f32>,
}

fn floats(p: &Path) -> Result<Vec<f32>> {
    let raw = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    anyhow::ensure!(raw.len() % 4 == 0, "{}: not a whole number of f32", p.display());
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub struct Bundle {
    pub dir: PathBuf,
    pub meta: Pipeline,
    pub tables: Tables,
}

impl Bundle {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let meta: Pipeline = serde_json::from_str(
            &std::fs::read_to_string(dir.join("pipeline.json")).with_context(|| {
                format!(
                    "no pipeline.json in {} — run `uv run pitvis-export` first",
                    dir.display()
                )
            })?,
        )?;
        let b = dir.join("bin");
        let tables = Tables {
            pe: floats(&b.join("pe.bin"))?,
            phase: floats(&b.join("phase.bin"))?,
            steps_mean: floats(&b.join("steps_mean.bin"))?,
            steps_std: floats(&b.join("steps_std.bin"))?,
            inst_mean: floats(&b.join("inst_mean.bin"))?,
            inst_std: floats(&b.join("inst_std.bin"))?,
            inst_tau: floats(&b.join("inst_tau.bin"))?,
        };
        // The two heads must read one feature space: a frame is embedded once
        // per pass, so a mismatch would mean serving two different encoders.
        anyhow::ensure!(
            meta.steps.feature_dim == meta.instruments.feature_dim
                && meta.steps.feature_dim == meta.backbone.feature_dim,
            "feature-dim mismatch across the bundle"
        );
        Ok(Self { dir, meta, tables })
    }

    pub fn graph(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{name}.onnx"))
    }
}
