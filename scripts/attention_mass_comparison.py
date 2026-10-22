#!/usr/bin/env python3
"""Measure class-token attention mass for the control and the proposed model.

Why this exists
---------------
The manuscript's attention claim currently rests on ``history.csv``: the AMG runs
record ``attention_loc`` (which is ``-log M``), and epoch 0 of those runs averages
to a mass of 0.160. That number describes the *first epoch of AMG training*, not an
untrained encoder, because a full epoch of gradient steps with the AMG term active
has already happened by the time it is written. It also says nothing about the
control, whose configuration sets ``lambda_attention = 0`` and therefore never
computes or logs the term at all.

The stronger statement -- that training *without* attention supervision leaves the
class token off the lesion, so AMG is not merely accelerating something the model
would have reached anyway -- needs the trained control's mass measured directly from
its weights. Figure 2 already does this, but only for the three slices it displays.
This script does it for every held-out slice of the fold.

What it measures
----------------
For each held-out slice of one fold, under both checkpoints:

* ``mass``     -- fraction of final-block class-token attention inside the lesion box.
* ``chance``   -- the same quantity under uniform attention, which equals the mean of
                  the box mask pooled to the patch grid. This is the per-slice version
                  of the paper's global 0.20 figure, so "below chance" becomes exact
                  per slice rather than an appeal to a cohort median.
* ``gain``     -- proposed minus control.

Mass is computed by ``ppgl.experiments.figures._patch_mass``, the same function the
comparison figure uses, which in turn reproduces ``AttentionLocalisationLoss``. It is
imported rather than reimplemented on purpose: a second implementation here could
silently disagree with the objective that was actually optimised.

Usage
-----
    python scripts/attention_mass_comparison.py \
        --output-dir results_ckpt \
        --proposed-checkpoint results_ckpt/runs/amg/folds/r0f0/model.pt \
        --control-checkpoint  results_ckpt/runs/amg_control/folds/r0f0/model.pt \
        --csv results_ckpt/attention_mass_r0f0.csv

Scope
-----
One fold. The checkpoints exist for r0f0 only, because a ViT-B/16 state dict is
~340 MB and the full grid is over a thousand folds. Any claim drawn from this is a
single-fold claim and should be worded as one. To cover all 50 folds, log the term in
the control arm instead of persisting 50 further checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as stats
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _summarise(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": stats.mean(values),
        "sd": stats.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": stats.median(values),
    }


def _fmt(label: str, s: Dict[str, float]) -> str:
    if not s.get("n"):
        return f"  {label:<26s} (no rows)"
    return (
        f"  {label:<26s} n={s['n']:<4d} mean={s['mean']:.4f} "
        f"sd={s['sd']:.4f} median={s['median']:.4f} "
        f"range=[{s['min']:.3f}, {s['max']:.3f}]"
    )


#: Candidate patient-identity columns, in preference order. The manifest uses
#: ``patient_uid``; the others are accepted so a schema change degrades loudly via
#: :func:`_patient_key` rather than silently collapsing every slice into its own
#: "patient" and making the patient-level rows a copy of the slice-level ones.
_PATIENT_COLUMNS = ("patient_uid", "patient", "patient_key", "patient_id")


def _patient_key(frame, index: int) -> str:
    for column in _PATIENT_COLUMNS:
        if column in frame.columns:
            return str(frame.loc[index, column])
    raise KeyError(
        "no patient identity column found; expected one of "
        f"{_PATIENT_COLUMNS}, got {list(frame.columns)}"
    )


def _history_mass(run_dir: str) -> Optional[Dict[str, float]]:
    """First and last epoch mass from a run's history, when the term was logged."""
    path = os.path.join(run_dir, "history.csv")
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "attention_loc" not in rows[0]:
        return None
    return {
        "first": math.exp(-float(rows[0]["attention_loc"])),
        "last": math.exp(-float(rows[-1]["attention_loc"])),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", default="results_ckpt")
    parser.add_argument("--proposed", default="amg")
    parser.add_argument("--control", default="amg_control")
    parser.add_argument("--proposed-checkpoint")
    parser.add_argument("--control-checkpoint")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--csv", default=None, help="per-slice output CSV")
    args = parser.parse_args(argv)

    sys.path.insert(0, _repo_root())

    import torch

    from ppgl.data.dataset import PPGLSliceDataset
    from ppgl.experiments.data_prep import build_folds, prepare_frame

    # Private imports are deliberate: reusing the figure's own mass definition is the
    # point, so that this script cannot disagree with the figure or the loss.
    from ppgl.experiments.figures import _load_model_for_figure, _patch_mass

    runs = os.path.join(args.output_dir, "runs")
    ckpt_p = args.proposed_checkpoint or os.path.join(
        runs, args.proposed, "folds", f"r{args.repeat}f{args.fold}", "model.pt"
    )
    ckpt_c = args.control_checkpoint or os.path.join(
        runs, args.control, "folds", f"r{args.repeat}f{args.fold}", "model.pt"
    )
    for name, path in ((args.proposed, ckpt_p), (args.control, ckpt_c)):
        if not os.path.isfile(path):
            parser.error(f"no checkpoint for {name!r} at {path}")

    print(f"proposed : {args.proposed}  <- {ckpt_p}")
    print(f"control  : {args.control}  <- {ckpt_c}")

    cfg_p, model_p = _load_model_for_figure(args.output_dir, args.proposed, ckpt_p)
    cfg_c, model_c = _load_model_for_figure(args.output_dir, args.control, ckpt_c)

    # Refuse an unpaired comparison rather than reporting one.
    fp_p = torch.load(ckpt_p, map_location="cpu").get("fold_fingerprint")
    fp_c = torch.load(ckpt_c, map_location="cpu").get("fold_fingerprint")
    if fp_p and fp_c and fp_p != fp_c:
        print(
            f"ERROR: checkpoints are from different folds ({fp_p} vs {fp_c}); "
            "the comparison would not be paired",
            file=sys.stderr,
        )
        return 2
    print(f"fold fingerprint match: {bool(fp_p and fp_c and fp_p == fp_c)}")

    lam_p = getattr(cfg_p.loss, "lambda_attention", None)
    lam_c = getattr(cfg_c.loss, "lambda_attention", None)
    print(f"lambda_attention: proposed={lam_p}  control={lam_c}")
    if lam_c:
        print(
            "WARNING: the control has a non-zero attention weight; it is not a "
            "no-supervision arm",
            file=sys.stderr,
        )

    frame = prepare_frame(cfg_p)
    folds = build_folds(cfg_p, frame)
    selected = [f for f in folds if f.repeat == args.repeat and f.fold == args.fold]
    if not selected:
        print(f"ERROR: fold r{args.repeat}f{args.fold} not found", file=sys.stderr)
        return 2
    test_frame = frame.iloc[selected[0].test_idx].reset_index(drop=True)

    dataset = PPGLSliceDataset(
        test_frame,
        cfg_p.data,
        train=False,
        mean=getattr(model_p, "mean", None),
        std=getattr(model_p, "std", None),
        return_mask=True,
    )

    import torch.nn.functional as F

    rows: List[Dict[str, Any]] = []
    skipped = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            mask = item.get("lesion_mask")
            if mask is None:
                skipped += 1
                continue
            mask_np = mask.detach().cpu().numpy()
            mask_np = mask_np[0] if mask_np.ndim == 3 else mask_np
            if mask_np.sum() <= 0:
                skipped += 1  # empty mask contributes nothing to AMG either
                continue

            image = item["image"].unsqueeze(0)
            out_p = model_p(image)
            out_c = model_c(image)
            if "attention_map" not in out_p or "attention_map" not in out_c:
                print(
                    "ERROR: a model returned no attention_map; is the control built "
                    "from the attn_lora family?",
                    file=sys.stderr,
                )
                return 2

            map_p = out_p["attention_map"][0].detach().cpu().numpy()
            map_c = out_c["attention_map"][0].detach().cpu().numpy()

            # Uniform attention gives mass equal to the mean of the pooled mask, so
            # this is the exact per-slice chance level.
            side = map_p.shape[-1]
            pooled = F.adaptive_avg_pool2d(
                torch.as_tensor(mask_np, dtype=torch.float32)[None, None], (side, side)
            )
            chance = float(pooled.mean())

            mass_p = _patch_mass(map_p, mask_np)
            mass_c = _patch_mass(map_c, mask_np)
            rows.append(
                {
                    "index": index,
                    "patient": _patient_key(test_frame, index),
                    "cluster": str(test_frame.loc[index, "cluster"]),
                    "chance": chance,
                    "mass_control": mass_c,
                    "mass_proposed": mass_p,
                    "gain": mass_p - mass_c,
                }
            )

    if not rows:
        print("ERROR: no held-out slice produced a usable mask", file=sys.stderr)
        return 2

    control = [r["mass_control"] for r in rows]
    proposed = [r["mass_proposed"] for r in rows]
    chance = [r["chance"] for r in rows]
    gains = [r["gain"] for r in rows]

    print(f"\nheld-out slices scored: {len(rows)} (skipped {skipped} without a mask)")
    print(f"fold: r{args.repeat}f{args.fold}\n")

    print("SLICE-LEVEL ATTENTION MASS INSIDE THE LESION BOX")
    print(_fmt("uniform-attention chance", _summarise(chance)))
    print(_fmt(f"control (lambda=0)", _summarise(control)))
    print(_fmt(f"proposed (lambda={lam_p})", _summarise(proposed)))
    print(_fmt("gain (proposed - control)", _summarise(gains)))

    below_c = sum(1 for r in rows if r["mass_control"] < r["chance"])
    below_p = sum(1 for r in rows if r["mass_proposed"] < r["chance"])
    wins = sum(1 for g in gains if g > 0)
    print(
        f"\n  control below its own chance level : {below_c}/{len(rows)}"
        f"\n  proposed below its own chance level: {below_p}/{len(rows)}"
        f"\n  slices where proposed > control    : {wins}/{len(rows)}"
    )

    print("\nPER CLUSTER (slice level)")
    by_cluster: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cluster[r["cluster"]].append(r)
    for cluster in sorted(by_cluster):
        group = by_cluster[cluster]
        print(
            f"  {cluster:<12s} n={len(group):<3d} "
            f"chance={stats.mean(x['chance'] for x in group):.3f}  "
            f"control={stats.mean(x['mass_control'] for x in group):.3f}  "
            f"proposed={stats.mean(x['mass_proposed'] for x in group):.3f}  "
            f"gain={stats.mean(x['gain'] for x in group):+.3f}"
        )

    # Patient-level, since slices from one patient are not independent.
    by_patient: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient"]].append(r)
    pat_c = [stats.mean(x["mass_control"] for x in g) for g in by_patient.values()]
    pat_p = [stats.mean(x["mass_proposed"] for x in g) for g in by_patient.values()]
    pat_ch = [stats.mean(x["chance"] for x in g) for g in by_patient.values()]
    print(
        f"\nPATIENT-LEVEL (mean over each patient's slices; "
        f"{len(by_patient)} patients, {len(rows)} slices)"
    )
    print(_fmt("uniform-attention chance", _summarise(pat_ch)))
    print(_fmt("control", _summarise(pat_c)))
    print(_fmt("proposed", _summarise(pat_p)))
    print(
        f"  control below chance: "
        f"{sum(1 for c, h in zip(pat_c, pat_ch) if c < h)}/{len(pat_c)}"
        f"   proposed above control: "
        f"{sum(1 for p, c in zip(pat_p, pat_c) if p > c)}/{len(pat_c)}"
    )

    print("\nTRAINING-HISTORY CROSS-CHECK (this fold only)")
    for name in (args.proposed, args.control):
        hist = _history_mass(
            os.path.join(runs, name, "folds", f"r{args.repeat}f{args.fold}")
        )
        if hist is None:
            print(f"  {name:<12s} attention_loc not logged (expected when lambda=0)")
        else:
            print(
                f"  {name:<12s} epoch 0 mass={hist['first']:.4f}  "
                f"final epoch mass={hist['last']:.4f}"
            )

    print(
        "\nNote: no p-value is reported. Slices from the same patient are correlated, "
        "so a per-slice paired test would overstate confidence, and a single fold "
        "cannot feed the corrected resampled t-test used elsewhere in the paper."
    )

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        fields = ["index", "patient", "cluster", "chance", "mass_control", "mass_proposed", "gain"]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in fields})
        print(f"\nper-slice values written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
