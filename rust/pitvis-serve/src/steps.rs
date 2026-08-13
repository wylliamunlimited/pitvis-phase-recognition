//! Step recognition: ARST memory, then the auto-regressive decode.
//!
//! THIS IS A PORT, NOT A DESIGN. The reference is `rollout()` in
//! `src/pitvis/inference/export.py`, which is itself a mirror of `cci_decode`
//! in `training/arst.py`. Anything that differs here is a bug, and
//! `pitvis-export --verify` is the gate that says so: the bar is agreement on
//! every second, not on tensors.
//!
//! Three behaviours that look optional and are not:
//!   * masking 0/11/13 out of the argmax, when the checkpoint was trained that
//!     way — ignoring the tag discards most of the winner's advantage;
//!   * the CCI probe, which asserts the OLD phase forward and only accepts a
//!     transition the decoder still wants;
//!   * the trailing-window offset, which is why the positional slice starts at
//!     `lo` and not at 0.

use crate::bundle::Bundle;
use anyhow::Result;
use ort::session::Session;
use ort::value::TensorRef;

pub struct StepModel {
    front: Session,
    decode: Session,
}

impl StepModel {
    pub fn load(b: &Bundle) -> Result<Self> {
        Ok(Self {
            front: Session::builder()?.commit_from_file(b.graph("front"))?,
            decode: Session::builder()?.commit_from_file(b.graph("decode"))?,
        })
    }

    /// (T, D) raw features -> (1, T, d_model) memory, standardised on the way in.
    pub fn memory(&mut self, b: &Bundle, feats: &[f32], t: usize) -> Result<Vec<f32>> {
        let d = b.meta.steps.feature_dim;
        let mut x = vec![0f32; t * d];
        for i in 0..t * d {
            x[i] = (feats[i] - b.tables.steps_mean[i % d]) / b.tables.steps_std[i % d];
        }
        let out = self
            .front
            .run(ort::inputs!["x" => TensorRef::from_array_view(([t, d], x.as_slice()))?])?;
        // Outputs by INDEX, not by name. The dynamo exporter names an output
        // after the last op that produced it ("layer_norm_1", "select_6"),
        // which is an implementation detail of the graph and would move if the
        // model changed. Inputs keep their names because those come from the
        // exported forward() signature and are meaningful.
        let (_, data) = out[0].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    /// Logits at position `t`, given the decoder's own past labels `prev`.
    fn logits(&mut self, b: &Bundle, mem: &[f32], t: usize, prev: &[usize]) -> Result<Vec<f32>> {
        let s = &b.meta.steps;
        let (dm, w) = (s.d_model, s.width);
        let lo = t.saturating_sub(w);
        let l = t + 1 - lo;

        // y = phase[prev[lo..=t]] + pe[lo..=t] — the decoder input, assembled
        // here rather than in the graph so no dynamic slice has to be exported.
        let mut y = vec![0f32; l * dm];
        for (k, &p) in prev[lo..=t].iter().enumerate() {
            for j in 0..dm {
                y[k * dm + j] = b.tables.phase[p * dm + j] + b.tables.pe[(lo + k) * dm + j];
            }
        }
        let mem_win = &mem[lo * dm..(t + 1) * dm];

        let out = self.decode.run(ort::inputs![
            "mem_win" => TensorRef::from_array_view(([1usize, l, dm], mem_win))?,
            "y_win"   => TensorRef::from_array_view(([1usize, l, dm], y.as_slice()))?
        ])?;
        let (_, data) = out[0].try_extract_tensor::<f32>()?;
        let mut lg = data.to_vec();
        if s.mask_excluded {
            for &c in &s.excluded {
                lg[c as usize] = f32::NEG_INFINITY;
            }
        }
        Ok(lg)
    }

    /// The full decode. Mirrors `export.rollout` line for line.
    pub fn decode_video(&mut self, b: &Bundle, mem: &[f32], t: usize) -> Result<Vec<usize>> {
        let s = &b.meta.steps;
        let mut preds = vec![0usize; t];
        let mut prev = vec![s.sos; t + s.cci_n + 1];

        for i in 0..t {
            let mut p = argmax(&self.logits(b, mem, i, &prev)?);
            if i > 0 && p != preds[i - 1] {
                // The consistency constraint: hold the previous phase forward
                // and only accept the transition if the decoder still wants it
                // at every one of the next cci_n frames.
                let mut probe = prev.clone();
                let mut accept = true;
                for j in 1..=s.cci_n {
                    if i + j >= t {
                        break;
                    }
                    probe[i + j] = preds[i - 1];
                    if argmax(&self.logits(b, mem, i + j, &probe)?) != p {
                        accept = false;
                        break;
                    }
                }
                if !accept {
                    p = preds[i - 1];
                }
            }
            preds[i] = p;
            if i + 1 < prev.len() {
                prev[i + 1] = p;
            }
        }
        Ok(preds)
    }
}

/// First maximum, matching numpy's `argmax` tie-breaking.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}
