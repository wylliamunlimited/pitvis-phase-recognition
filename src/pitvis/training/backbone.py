"""Fine-tune the backbone on PitVis frames — a surgical-specific encoder.

WHY. Everything the repo has done so far rides a FROZEN encoder that has never
seen an endoscope: ImageNet ResNet-50, or DINOv2 trained on web images. Both
our reproductions sit near half their published Table 8 benchmark, and both
papers fine-tune the backbone on the surgical data before freezing it (ARST
§3.3 trains ResNet-50 for 50 epochs; SANO's is "the trained CNN was frozen").
That step is the one we have never been able to take, because extraction threw
the pixels away.

THE EVIDENCE THAT IT IS THE BINDING CONSTRAINT. A per-class average-precision
probe on frozen DINOv2 features, one-vs-rest:

    tissue glue          282 train positives    AP 0.767
    micro doppler        679                    AP 0.731
    cup forceps        1,635                    AP 0.055
    retractable knife    492                    AP 0.015
    bipolar forceps      184                    AP 0.026

Rarity does not predict difficulty — tissue glue is rarer than four of the weak
classes and is nearly separable. What predicts it is whether the encoder can
SEE the instrument, and for six of nineteen it cannot. No threshold, class
weight or sampler recovers information that is not in the features.

WHAT THIS TRAINS. One backbone, two heads: 15-way steps (cross-entropy) and
19-way instruments (BCE). Every annotation row carries both labels, so the
second head is free supervision, and it is what SANO's own design does in
miniature. A single encoder serving both tasks also means one fine-tune rather
than two, and one feature space downstream rather than two.

AUGMENTATION IS FINALLY POSSIBLE. Random crop, horizontal flip and colour
jitter are three of the rows our faithfulness tables mark as not reproduced,
purely because there were no pixels to augment. The frame cache stores 384px
squares so a 224 crop has real headroom.

COST, measured on MPS: ResNet-50 trains at 96 img/s, so one epoch over the
84,666 training frames is ~15 min and the papers' 50 epochs is ~12 h. DINOv2
ViT-B is 29 img/s — 41 h — which is why the backbone with the better frozen
features is the worse one to fine-tune here. Start with a short pilot and
re-run the AP probe before committing to the long run.

Usage:
    uv run pitvis-finetune --epochs 5              # the pilot
    uv run pitvis-finetune --epochs 50 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pitvis.data.dataset import NUM_CLASSES, TRAIN, VAL
from pitvis.data.extract_frames import DEFAULT_SIZE, video_frames
from pitvis.evaluation.instruments import NUM_INSTRUMENTS, multihot
from pitvis.paths import DATA, RAW

CKPT_BACKBONE = DATA / "backbone"
CROP = 224


def annotations(vid: int, expected: int) -> tuple[np.ndarray, np.ndarray]:
    """(steps (T,) encoded 0..14, instruments (T, 19) multi-hot) for one video."""
    import pandas as pd
    df = pd.read_csv(RAW / f"annotations_{vid:02d}.csv")
    steps = df["int_step"].to_numpy()[:expected].copy()
    steps[steps == -1] = 0
    inst = df[["int_instrument1", "int_instrument2"]].to_numpy()[:expected]
    return steps.astype(np.int64), multihot(inst.astype(np.int64))


class Frames(Dataset):
    """JPEG frames plus both label sets.

    Reads from disk rather than holding pixels in memory: 84,666 frames at
    384px is ~37 GB uncompressed, and the whole point of the JPEG cache is that
    the working set stays on disk while only a batch is resident.
    """

    def __init__(self, videos: list[int], size: int, train: bool):
        from torchvision import transforms as T

        self.items: list[tuple[Path, int, np.ndarray]] = []
        for vid in videos:
            d = video_frames(size, vid)
            paths = sorted(d.glob("*.jpg"))
            if not paths:
                raise SystemExit(
                    f"no frames at {d} — run `uv run pitvis-frames --size {size}`"
                )
            steps, inst = annotations(vid, len(paths))
            for i, p in enumerate(paths):
                self.items.append((p, int(steps[i]), inst[i]))

        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            # The three rows our faithfulness tables marked "we cannot" — not
            # because they were rejected, but because there were no pixels.
            self.tf = T.Compose([
                T.RandomResizedCrop(CROP, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.ToTensor(), norm,
            ])
        else:
            self.tf = T.Compose([T.Resize(CROP), T.CenterCrop(CROP), T.ToTensor(), norm])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, step, inst = self.items[i]
        x = self.tf(Image.open(path).convert("RGB"))
        return x, step, torch.from_numpy(inst.astype(np.float32))


class MultiTask(nn.Module):
    """Backbone plus two heads. `backbone` alone is what extraction reuses.

    `model_kwargs` reaches `timm.create_model` untouched, and for a ViT it is
    not optional. DINOv2's weights ship at 518x518, so without `img_size=224`
    the model is built for a 37x37 patch grid and rejects the 224 tensor this
    dataset produces — "Input height (518) doesn't match model (224)", the same
    trap extraction hit. It must also match the `img_size` of the space that
    will later read this checkpoint, or the encoder is fine-tuned at one
    resolution and inferred at another.
    """

    def __init__(self, name: str, pretrained: bool = True, **model_kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(name, pretrained=pretrained,
                                          num_classes=0, **model_kwargs)
        d = self.backbone.num_features
        self.steps = nn.Linear(d, NUM_CLASSES)
        self.instruments = nn.Linear(d, NUM_INSTRUMENTS)

    def forward(self, x):
        h = self.backbone(x)
        return self.steps(h), self.instruments(h)


def device_of(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(ds: Frames, cap: float) -> torch.Tensor:
    counts = np.bincount([s for _, s, _ in ds.items], minlength=NUM_CLASSES)
    inv = np.where(counts > 0, len(ds) / (NUM_CLASSES * np.maximum(counts, 1)), 1.0)
    return torch.from_numpy(np.clip(inv, 1.0, cap)).float()


def pos_weight(ds: Frames, cap: float) -> torch.Tensor:
    Y = np.stack([i for _, _, i in ds.items])
    pos = Y.sum(0).astype(np.float64)
    w = np.where(pos > 0, (len(Y) - pos) / np.maximum(pos, 1), 1.0)
    return torch.from_numpy(np.clip(w, 1.0, cap)).float()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbone", default="resnet50",
                    help="timm model to fine-tune (default: resnet50 — 4x "
                         "cheaper to train than ViT-B and what both papers used)")
    ap.add_argument("--img-size", type=int, default=None, metavar="PX",
                    help="build the backbone for this input size. REQUIRED for "
                         "a ViT — DINOv2 ships at 518 and rejects the 224 crop "
                         "this dataset produces unless told otherwise. Leave "
                         "unset for a CNN, which infers its size from the input")
    ap.add_argument("--epochs", type=int, default=5,
                    help="5 is the pilot; the papers use 50")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inst-weight", type=float, default=1.0,
                    help="weight on the instrument head's loss")
    ap.add_argument("--weight-cap", type=float, default=10.0)
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--device", default=None, choices=("cuda", "mps", "cpu"))
    ap.add_argument("--tag", default=None,
                    help="output directory name (default: <backbone>-<epochs>ep)")
    ap.add_argument("--exclude", type=int, nargs="*", default=None, metavar="V",
                    help="video ids to hold OUT of fine-tuning. Required for a "
                         "per-fold backbone: a cross-validation fold is only "
                         "honest if the encoder never saw its held-out videos")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap training frames — for smoke tests only")
    args = ap.parse_args(argv)

    dev = device_of(args.device)
    tag = args.tag or f"{args.backbone.split('.')[0]}-{args.epochs}ep"
    out = CKPT_BACKBONE / tag
    out.mkdir(parents=True, exist_ok=True)

    excluded = set(args.exclude or [])
    videos = [v for v in TRAIN if v not in excluded]
    if excluded:
        print(f"holding out {sorted(excluded)} — this backbone never sees them")
    train_ds = Frames(videos, args.size, train=True)
    val_ds = Frames(VAL, args.size, train=False)
    if args.limit:
        train_ds.items = train_ds.items[:args.limit]
    print(f"device {dev}  backbone {args.backbone}  epochs {args.epochs}")
    print(f"train {len(train_ds):,} frames from {len(videos)} videos   "
          f"val {len(val_ds):,} from {len(VAL)}")

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=(dev.type == "cuda"),
                        persistent_workers=args.workers > 0, drop_last=True)

    # Fail here, loudly, rather than after an epoch of training: a ViT built at
    # its native 518 accepts nothing this dataset produces, and the error is
    # far less obvious once it surfaces inside a DataLoader worker.
    kwargs = {"img_size": args.img_size} if args.img_size else {}
    model = MultiTask(args.backbone, **kwargs).to(dev)
    with torch.no_grad():
        probe = model.backbone(torch.zeros(1, 3, CROP, CROP, device=dev))
    print(f"backbone accepts {CROP}x{CROP} -> {probe.shape[1]}-d")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(weight=class_weights(train_ds, args.weight_cap).to(dev))
    bce = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight(train_ds, args.pos_weight_cap).to(dev))

    # Resume state, written after every epoch.
    #
    # WHY EPOCH GRANULARITY. On spot capacity the unit of loss should be the
    # unit of work you can afford to repeat. Without this the unit is a whole
    # encoder — preempted at epoch 45 of 50 and four hours are gone — and no
    # amount of bucket plumbing helps, because the thing that was lost was
    # never written down. With it, the worst case is the epoch in flight.
    #
    # The optimiser and scheduler are saved with the weights on purpose.
    # AdamW carries first and second moments per parameter, and cosine decay
    # carries its position; restoring weights alone would resume at the wrong
    # learning rate with the moments zeroed, which is a different training run
    # that happens to start from the same numbers.
    resume_path = out / "resume.pt"
    start_ep = 0
    if resume_path.exists():
        r = torch.load(resume_path, map_location=dev, weights_only=False)
        model.load_state_dict(r["model"])
        opt.load_state_dict(r["opt"])
        sched.load_state_dict(r["sched"])
        start_ep = r["epoch"]
        print(f"resuming from {resume_path} at epoch {start_ep + 1}/{args.epochs}")
        if r.get("trained_on") != sorted(videos):
            raise SystemExit(
                f"resume state was trained on {r.get('trained_on')}, this run "
                f"holds out a different set. Delete {resume_path} to start over "
                f"— resuming would silently mix two different encoders.")

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        run = correct = seen = 0.0
        te = time.time()
        for n, (x, s, i) in enumerate(loader, 1):
            x, s, i = x.to(dev, non_blocking=True), s.to(dev), i.to(dev)
            ls, li = model(x)
            loss = ce(ls, s) + args.inst_weight * bce(li, i)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(s)
            correct += (ls.argmax(1) == s).sum().item()
            seen += len(s)
            if n % 200 == 0:
                print(f"  ep{ep + 1} {n * args.batch:,}/{len(train_ds):,}  "
                      f"loss {run / seen:.4f}  step-acc {correct / seen:.3f}  "
                      f"({seen / (time.time() - te):.0f} img/s)")
        sched.step()
        print(f"epoch {ep + 1}/{args.epochs}: loss {run / seen:.4f}  "
              f"step-acc {correct / seen:.3f}  [{time.time() - te:.0f}s]")

        # Write to a temp file and rename. A preemption lands mid-write often
        # enough to matter over 50 epochs, and a half-written resume file is
        # worse than none: it fails to load at the start of the NEXT run,
        # after the instance has already been paid for.
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"epoch": ep + 1, "model": model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "trained_on": sorted(videos)}, tmp)
        tmp.replace(resume_path)

    # Only the backbone matters downstream — the heads exist to shape it.
    # `trained_on` is what lets crossval refuse to hold out a video this
    # encoder has already memorised. Without it the leak is invisible.
    torch.save({"backbone": model.backbone.state_dict(), "name": args.backbone,
                "trained_on": sorted(videos), "args": vars(args)},
               out / "backbone.pt")
    (out / "result.json").write_text(json.dumps(
        {"backbone": args.backbone, "epochs": args.epochs,
         "train_frames": len(train_ds), "trained_on": sorted(videos),
         "seconds": round(time.time() - t0, 1), "args": vars(args)},
        indent=2) + "\n")
    # The run finished, so the resume state is now dead weight — and it is
    # optimiser-sized, ~3x the encoder. Leaving it would upload a gigabyte of
    # scratch per fold and make a completed encoder look like an interrupted one.
    resume_path.unlink(missing_ok=True)
    print(f"\nwrote {out / 'backbone.pt'}  [{(time.time() - t0) / 60:.1f} min]")
    print(f"next: add a space pointing at it, then `uv run pitvis-extract "
          f"--space <name>`")


if __name__ == "__main__":
    main()
