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

COST, and why this trains in bf16. Measured on MPS: ResNet-50 at 96 img/s, so
one epoch over the 84,666 training frames is ~15 min. Measured on an L4 in
fp32: DINOv2 ViT-B at **27 img/s** — one epoch took 52 minutes and 50 epochs
projected to **~43 hours**, against an 8-hour instance cap.

That was not the dataloader. The CPU sat 79% idle while the GPU ran flat out,
which is the signature of arithmetic rather than input starvation: an L4's
throughput lives in its bf16 tensor cores and fp32 uses almost none of it.
Hence the autocast below. `--no-amp` restores fp32 for isolating a suspected
precision problem, and is otherwise the slow path.

The curve is steep early — the 5-epoch ResNet-50 pilot already moved mean AP
from 0.271 to 0.445 — so prefer a short run and the AP probe over a long one
taken on faith.

Usage:
    uv run pitvis-finetune --epochs 5              # the pilot
    uv run pitvis-finetune --epochs 50 --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
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


def depth_of(name: str, n_blocks: int) -> int:
    """Which layer a backbone parameter belongs to, 0 = stem, n = deepest.

    Name-based rather than structural, because the two families we fine-tune
    expose their depth differently and a structural walk would have to special
    case both anyway. Anything unrecognised — the final norm, a pooling head —
    is treated as the TOP of the backbone, which is the conservative choice:
    it gets the largest learning rate, so an unknown parameter is trained
    rather than silently frozen.
    """
    m = re.search(r"blocks\.(\d+)\.", name)          # timm ViT
    if m:
        return int(m.group(1)) + 1
    m = re.search(r"layer(\d+)\.", name)             # timm / torchvision ResNet
    if m:
        return int(m.group(1))
    if any(k in name for k in ("patch_embed", "cls_token", "pos_embed",
                               "conv1", "bn1", "stem")):
        return 0
    return n_blocks


def layer_decay_groups(model: nn.Module, backbone_lr: float, head_lr: float,
                       decay: float) -> list[dict]:
    """Parameter groups with layer-wise learning-rate decay.

    WHY. Fine-tuning DINOv2 at a uniform 1e-4 destroyed it — mean AP fell
    0.350 -> 0.270 with 16 of 19 classes worse. A single rate treats the
    patch embedding, which encodes generic visual structure worth keeping, the
    same as the last block, which is the part that should specialise. Layer-wise
    decay makes early layers move `decay^k` times as fast as late ones, which is
    the standard prescription for adapting a strong self-supervised encoder.

    With decay=1.0 this degenerates to a single group at `backbone_lr`, so the
    old behaviour is still reachable and reproducible.
    """
    depths = {n: depth_of(n, 0) for n, _ in model.backbone.named_parameters()}
    n_blocks = max(depths.values(), default=0)
    depths = {n: depth_of(n, n_blocks) for n in depths}

    groups: dict[int, list] = {}
    for n, p in model.backbone.named_parameters():
        if p.requires_grad:
            groups.setdefault(depths[n], []).append(p)

    out = [{"params": ps, "lr": backbone_lr * decay ** (n_blocks - d),
            "name": f"backbone.depth{d}"}
           for d, ps in sorted(groups.items())]
    # The heads are new and random — they get the full rate regardless.
    out.append({"params": list(model.steps.parameters())
                + list(model.instruments.parameters()),
                "lr": head_lr, "name": "heads"})
    return out


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
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate for the randomly-initialised HEADS")
    ap.add_argument("--backbone-lr", type=float, default=None, metavar="LR",
                    help="learning rate for the pretrained encoder. Defaults to "
                         "lr/10. A uniform 1e-4 across a ViT-B destroyed DINOv2 "
                         "(mean AP 0.350 -> 0.270), which is what this exists to "
                         "avoid")
    ap.add_argument("--layer-decay", type=float, default=0.75, metavar="D",
                    help="layer-wise lr decay: layer k trains at "
                         "backbone_lr * D^(depth-k). 1.0 disables it")
    ap.add_argument("--warmup", type=int, default=200, metavar="STEPS",
                    help="linear warmup steps. ViTs are unusually sensitive to "
                         "large updates in the first few hundred steps")
    ap.add_argument("--val-videos", type=int, default=3, metavar="N",
                    help="videos held out of TRAIN for early stopping. NOT the "
                         "VAL split — stopping on VAL would be selection on VAL")
    ap.add_argument("--patience", type=int, default=3, metavar="EPOCHS",
                    help="stop after this many epochs without a better "
                         "validation loss. 0 disables early stopping")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inst-weight", type=float, default=1.0,
                    help="weight on the instrument head's loss")
    ap.add_argument("--weight-cap", type=float, default=10.0)
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--no-amp", action="store_true",
                    help="disable bf16 autocast and train in fp32. Only for "
                         "isolating a suspected precision problem — it is ~5x "
                         "slower on an L4")
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
    # EARLY-STOPPING SPLIT, CARVED FROM TRAIN — never from VAL.
    #
    # The previous version built `Frames(VAL, ...)` and never used it, so 50
    # epochs ran with nothing watching and DINOv2 was destroyed without a
    # signal. Wiring VAL in would have been worse: stopping on VAL is selection
    # on VAL, and it would contaminate the single VAL scoring the whole
    # protocol is built around.
    #
    # Held out BY VIDEO, not by frame. Frames within a video are ~69 to a step
    # segment and look alike, so a frame-level split would put near-duplicates
    # on both sides and report a validation loss that only measures memory.
    # Taken from the END of the list so `--exclude` (per-fold encoders) still
    # removes videos from the front deterministically.
    holdout = videos[-args.val_videos:] if args.val_videos else []
    fit_videos = [v for v in videos if v not in holdout]

    train_ds = Frames(fit_videos, args.size, train=True)
    val_ds = Frames(holdout, args.size, train=False) if holdout else None
    if args.limit:
        train_ds.items = train_ds.items[:args.limit]
        if val_ds:
            val_ds.items = val_ds.items[:args.limit]
    print(f"device {dev}  backbone {args.backbone}  epochs {args.epochs}")
    print(f"train {len(train_ds):,} frames from {len(fit_videos)} videos")
    if val_ds:
        print(f"early-stop split: {len(val_ds):,} frames from videos {holdout} "
              f"(carved from TRAIN; VAL is untouched)")
    else:
        print("early-stop split: NONE (--val-videos 0) — nothing will stop this run")

    val_loader = (DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers,
                             pin_memory=(dev.type == "cuda"))
                  if val_ds else None)

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
    # MIXED PRECISION. Measured on an L4: fp32 ViT-B ran at ~27 img/s, which is
    # roughly 5% of what the card should manage, with the CPU 79% idle — so it
    # was neither the dataloader nor the disk. An L4's throughput lives in its
    # bf16 tensor cores, and fp32 forgoes essentially all of it. At 27 img/s,
    # 50 epochs over 84,666 frames is ~43 hours against an 8-hour cap.
    #
    # bfloat16 rather than float16, and therefore NO GradScaler: bf16 keeps
    # fp32's exponent range, so the underflow that fp16 needs loss scaling to
    # survive does not arise. One fewer moving part in a loop that has to
    # resume correctly after a preemption.
    #
    # Master weights stay fp32 — autocast only casts the ops it runs — so the
    # optimiser state, and therefore resume.pt, is unchanged in dtype and
    # remains compatible with checkpoints written before this.
    amp_ok = dev.type in ("cuda", "cpu") and not args.no_amp
    if amp_ok and dev.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("bf16 unsupported on this GPU — falling back to fp32")
        amp_ok = False
    print(f"mixed precision: {'bf16' if amp_ok else 'off (fp32)'}", flush=True)

    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr / 10
    groups = layer_decay_groups(model, backbone_lr, args.lr, args.layer_decay)
    print(f"lr: heads {args.lr:.1e}  backbone {backbone_lr:.1e} "
          f"(layer decay {args.layer_decay}, {len(groups)} groups, "
          f"deepest->stem {groups[-2]['lr']:.1e}->{groups[0]['lr']:.1e})")
    opt = torch.optim.AdamW(groups, lr=args.lr)

    # Warmup then cosine, stepped PER BATCH. ViTs are unusually sensitive to
    # large updates in the first few hundred steps, and a strong pretrained
    # encoder has the most to lose from them.
    steps_per_epoch = max(1, len(train_ds) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warm = min(args.warmup, max(1, total_steps - 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warm))],
        milestones=[warm])
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

    best_val, since_best, stopped_early = float("inf"), 0, False
    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        run = correct = seen = 0.0
        te = time.time()
        for n, (x, s, i) in enumerate(loader, 1):
            x, s, i = x.to(dev, non_blocking=True), s.to(dev), i.to(dev)
            # Only the forward and the losses are autocast. The backward runs
            # in whatever dtype each op recorded, and the optimiser step stays
            # fp32 — putting opt.step() inside the context is the classic way
            # to corrupt master weights.
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                enabled=amp_ok):
                ls, li = model(x)
                loss = ce(ls, s) + args.inst_weight * bce(li, i)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()          # per BATCH: warmup is measured in steps
            run += loss.item() * len(s)
            correct += (ls.argmax(1) == s).sum().item()
            seen += len(s)
            if n % 200 == 0:
                # flush=True because startup.sh pipes stdout through `tee`,
                # which makes it block-buffered: an hour of real training
                # showed zero progress lines, and the only honest signal was
                # resume.pt's mtime on disk. A progress line nobody can read
                # until the job ends is not a progress line.
                print(f"  ep{ep + 1} {n * args.batch:,}/{len(train_ds):,}  "
                      f"loss {run / seen:.4f}  step-acc {correct / seen:.3f}  "
                      f"({seen / (time.time() - te):.0f} img/s)", flush=True)
        print(f"epoch {ep + 1}/{args.epochs}: loss {run / seen:.4f}  "
              f"step-acc {correct / seen:.3f}  [{time.time() - te:.0f}s]", flush=True)

        # THE SIGNAL THAT WAS MISSING. Held-out loss on TRAIN videos the
        # encoder never saw, so it measures adaptation rather than memory.
        if val_loader is not None:
            model.eval()
            vl = vn = vcorrect = 0.0
            with torch.no_grad():
                for x, sT, iT in val_loader:
                    x, sT, iT = x.to(dev), sT.to(dev), iT.to(dev)
                    with torch.autocast(device_type=dev.type,
                                        dtype=torch.bfloat16, enabled=amp_ok):
                        ls, li = model(x)
                        l = ce(ls, sT) + args.inst_weight * bce(li, iT)
                    vl += float(l) * len(sT)
                    vcorrect += float((ls.argmax(1) == sT).sum())
                    vn += len(sT)
            val_loss = vl / max(vn, 1)
            gap = (correct / seen) - (vcorrect / max(vn, 1))
            print(f"  val: loss {val_loss:.4f}  step-acc {vcorrect / max(vn,1):.3f}"
                  f"  (train-val acc gap {gap:+.3f})", flush=True)

            if val_loss < best_val - 1e-4:
                best_val, since_best = val_loss, 0
                torch.save({"backbone": model.backbone.state_dict(),
                            "name": args.backbone,
                            "trained_on": sorted(fit_videos), "epoch": ep + 1,
                            "val_loss": val_loss, "args": vars(args)},
                           out / "best.pt")
                print(f"  new best (epoch {ep + 1}) -> best.pt", flush=True)
            else:
                since_best += 1
                if args.patience and since_best >= args.patience:
                    print(f"  early stop: {since_best} epochs without "
                          f"improvement (best {best_val:.4f})", flush=True)
                    stopped_early = True

        # Write to a temp file and rename. A preemption lands mid-write often
        # enough to matter over 50 epochs, and a half-written resume file is
        # worse than none: it fails to load at the start of the NEXT run,
        # after the instance has already been paid for.
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"epoch": ep + 1, "model": model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "trained_on": sorted(videos)}, tmp)
        tmp.replace(resume_path)
        if stopped_early:
            break

    # BACKBONE.PT IS THE BEST EPOCH, NOT THE LAST ONE.
    #
    # Early stopping wrote best.pt and nothing read it: `spaces.py` loads
    # backbone.pt, so the encoder shipped downstream was the FINAL epoch — the
    # one with the worst validation loss, which is the exact weights early
    # stopping exists to discard. On run 2 that was epoch 5 (val 1.7758)
    # instead of epoch 2 (val 1.4812).
    #
    # Restoring here rather than repointing the space keeps backbone.pt the
    # single name every downstream reader already knows.
    best_epoch = None
    best_path = out / "best.pt"
    if best_path.exists():
        b = torch.load(best_path, map_location=dev, weights_only=False)
        model.backbone.load_state_dict(b["backbone"])
        best_epoch = b.get("epoch")
        print(f"restoring best epoch {best_epoch} (val {b.get('val_loss'):.4f}) "
              f"as backbone.pt", flush=True)

    torch.save({"backbone": model.backbone.state_dict(), "name": args.backbone,
                "trained_on": sorted(fit_videos), "best_epoch": best_epoch,
                "args": vars(args)},
               out / "backbone.pt")
    (out / "result.json").write_text(json.dumps(
        {"backbone": args.backbone, "epochs": args.epochs,
         "train_frames": len(train_ds), "trained_on": sorted(fit_videos),
         "best_epoch": best_epoch, "epochs_ran": ep + 1,
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
