"""Command-line interface.

    python -m ppgl.cli doctor                      # environment and data check
    python -m ppgl.cli inventory                   # build the manifest
    python -m ppgl.cli locations --review-csv ...   # provisional region labels
    python -m ppgl.cli folds                       # build and audit CV folds
    python -m ppgl.cli train --config configs/experiments/routed_soft.yaml
    python -m ppgl.cli grid --configs "configs/experiments/*.yaml"
    python -m ppgl.cli analyse --experiment routed_soft
    python -m ppgl.cli stats --reference routed_soft --others swin,vit,...
    python -m ppgl.cli tables --experiments ...
    python -m ppgl.cli figures --reference routed_soft

Any config value can be overridden from the command line::

    python -m ppgl.cli train --config ... --set optim.lr=3e-5 model.routing=anneal
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Sequence

from ppgl.config import Config, load_config
from ppgl.utils.logging import get_logger, setup_logging

log = get_logger("ppgl.cli")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _split_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_ints(value: Optional[str]) -> Optional[List[int]]:
    items = _split_list(value)
    return [int(item) for item in items] if items else None


def _expand_configs(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            if os.path.isfile(pattern):
                matched = [pattern]
            else:
                log.warning("config pattern %r matched nothing", pattern)
        # A leading underscore marks a shared fragment that other configs inherit
        # via _base_, not a runnable experiment. Without this, a directory glob
        # picks the fragment up and runs 50 folds of whatever the base defaults
        # happen to be, under a name nobody chose. Skipped only when it arrived
        # from a glob, so naming one explicitly still works.
        skipped = [p for p in matched if os.path.basename(p).startswith("_")]
        if skipped and len(matched) > 1:
            for path in skipped:
                log.info("skipping shared config fragment %s", path)
            matched = [p for p in matched if p not in skipped]
        paths.extend(matched)
    # De-duplicate while preserving order.
    seen, unique = set(), []
    for path in paths:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return unique


def _load(args) -> Config:
    cfg = load_config(getattr(args, "config", None), overrides=getattr(args, "set", None) or [])
    if getattr(args, "output_dir", None):
        cfg.output_dir = args.output_dir
    if getattr(args, "experiment", None):
        cfg.experiment = args.experiment
    if getattr(args, "smoke_test", False):
        cfg.smoke_test = True
        cfg.optim.epochs = min(cfg.optim.epochs, 2)
        cfg.cv.n_repeats = 1
        cfg.cv.n_splits = min(cfg.cv.n_splits, 3)
        cfg.cv.require_all_classes_per_fold = False
    return cfg


def _default_experiments(output_dir: str) -> List[str]:
    from ppgl.experiments import results as store

    return store.list_experiments(output_dir)


def _sync_label_space(cfg, output_dir: str, experiment: Optional[str] = None) -> str:
    """Align the process label space with the results being analysed.

    Analysis commands read predictions written by an earlier training run, whose
    class vocabulary is recorded in that run's ``config.json``. Trusting the
    command-line config instead would silently mislabel probability columns when
    the two disagree, so the stored value wins and a mismatch is reported.
    """
    from ppgl.experiments import results as store
    from ppgl.labels import set_active_label_space

    name = cfg.data.label_space
    if experiment:
        stored = (store.load_config_dict(output_dir, experiment).get("data") or {}).get(
            "label_space"
        )
        if stored and stored != name:
            log.info(
                "using label space %r recorded in %s/config.json (cli config said %r)",
                stored,
                experiment,
                name,
            )
            name = stored
    set_active_label_space(name)
    return name


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_doctor(args) -> int:
    """Report the execution environment and whether the data looks usable."""
    from ppgl.utils.device import describe_hardware, resolve_parallel_jobs

    cfg = _load(args)
    hardware = describe_hardware()
    print(json.dumps(hardware, indent=2, default=str))
    print(f"\nfold-parallel workers that would be used: {resolve_parallel_jobs(cfg.runtime.n_parallel_jobs)}")

    slices_dir = os.path.join(cfg.data.root, "Slices")
    xmls_dir = os.path.join(cfg.data.root, "XMLs")
    print(f"\ndata root: {os.path.abspath(cfg.data.root)}")
    print(f"  Slices/ present: {os.path.isdir(slices_dir)}")
    print(f"  XMLs/   present: {os.path.isdir(xmls_dir)}")
    print(f"  manifest: {cfg.data.manifest} (exists={os.path.isfile(cfg.data.manifest)})")

    # Pretrained weights. Checked here because a missing checkpoint otherwise
    # only surfaces once the grid has been dispatched to every GPU.
    from ppgl.models import weights as weights_module

    directory = weights_module.resolve_weights_dir(cfg.model.weights_dir)
    print(f"\nbundled pretrained weights: {directory}")
    report = weights_module.status(directory)
    for backbone, entry in report.items():
        marker = "OK     " if entry["present"] else ("MISSING" if entry["required"] else "absent ")
        size = f"{entry['size_mb']:.0f} MB" if entry["size_mb"] else f"~{entry['expected_mb']:.0f} MB"
        tag = "required" if entry["required"] else "optional"
        print(f"  {marker} {backbone:34s} {size:>9s}  [{tag}]  {entry['tag']}")

    absent_required = weights_module.missing(required_only=True, weights_dir=directory)
    if absent_required:
        print(
            f"\n  {len(absent_required)} REQUIRED checkpoint(s) missing: {', '.join(absent_required)}\n"
            "  Every deep model will fail until this is resolved. Either:\n"
            "    - unpack the archive's weights/ directory next to this project, or\n"
            "    - point at it:  --set model.weights_dir=/path/to/weights\n"
            "      (or export PPGL_WEIGHTS_DIR=/path/to/weights), or\n"
            "    - on a host with internet:  python -m ppgl.cli fetch-weights\n"
            "    - proceed WITHOUT pretrained weights (NOT publishable at n=630) via\n"
            "      --set model.allow_random_init=true\n"
            "  The tabular controls, including the location-only gate, need no weights."
        )
    else:
        print("\n  all required checkpoints present; the deep models can run fully offline")

    integrity = weights_module.verify(directory)
    bad = {name: state for name, state in integrity.items() if state not in ("ok",) and name != "_"}
    if bad:
        print(f"\n  WARNING integrity check failed for: {bad}")
    elif "_" not in integrity and integrity:
        print(f"  checksums verified for {len(integrity)} file(s)")

    if cfg.data.location_source == "heuristic":
        print(
            "\nWARNING: data.location_source='heuristic'. Anatomical region labels do not exist\n"
            "in this archive. Heuristic labels are provisional and must not be used for reported\n"
            "results. Supply radiologist labels via data.location_csv."
        )
    return 0


def cmd_fetch_weights(args) -> int:
    """Download the pretrained checkpoints into the bundled weights directory.

    Pure HTTPS, so this works on any machine with internet and needs neither
    torch nor timm installed.
    """
    from ppgl.models import weights as weights_module

    cfg = _load(args)
    directory = weights_module.resolve_weights_dir(cfg.model.weights_dir)

    if args.verify_only:
        results = weights_module.verify(directory)
        for name, state in results.items():
            print(f"  {state:20s} {name}")
        return 0 if all(state == "ok" for state in results.values()) else 1

    names = _split_list(args.backbones)
    print(f"target directory: {directory}\n")
    results = weights_module.download(
        backbones=names,
        weights_dir=directory,
        include_optional=bool(args.all),
        force=bool(args.force),
    )

    for name, state in results.items():
        marker = "OK     " if state in ("ok", "already present") else "FAILED "
        print(f"  {marker} {name:34s} {state}")

    failed = {name: state for name, state in results.items() if state.startswith("FAILED")}
    if failed:
        print(
            f"\n{len(failed)} of {len(results)} failed. If this host has no internet, run this "
            "command where it does and copy the directory across:\n"
            f"  rsync -a {directory}/ server:/path/to/ppgl_isbi2027/weights/"
        )
        return 1

    weights_module.write_checksums(directory)
    print("\nall requested checkpoints present; the deep models can run fully offline.")
    return 0


def cmd_inventory(args) -> int:
    from ppgl.data.inventory import build_manifest

    cfg = _load(args)
    build_manifest(
        cfg.data.root,
        output_csv=cfg.data.manifest,
        qa_json=os.path.join(cfg.output_dir, "data_qa.json"),
    )
    return 0


def cmd_clinical_template(args) -> int:
    """Emit the blank clinical covariate sheet, one row per patient."""
    from ppgl.data.clinical import SCHEMA, write_template
    from ppgl.data.inventory import load_manifest

    cfg = _load(args)
    frame = load_manifest(cfg.data.manifest)
    path = write_template(frame, args.out)

    print(f"\nwrote {frame['patient_uid'].nunique()} patient rows to {path}")
    print(f"legend: {os.path.splitext(path)[0]}_legend.txt\n")
    print("Columns to complete from the chart review:")
    for field in SCHEMA:
        allowed = f"  ({', '.join(field.categories)})" if field.categories else ""
        print(f"  {field.name:26s} [{field.kind}]{allowed}")
    print(
        "\nBlanks are fine: missing values are imputed per fold and flagged with a\n"
        "missing indicator, so a partially completed sheet is still usable. Then:\n"
        f"  ./run_all.sh --set data.clinical_csv={args.out}"
    )
    return 0


def cmd_ensemble(args) -> int:
    """Combine stored per-fold probabilities into a new pseudo-experiment.

    Costs no GPU time: it reads predictions the grid already wrote. The result is
    written back into the results store so ``stats``, ``tables`` and ``figures``
    treat the ensemble as one more model.
    """
    from ppgl.experiments.ensemble import EnsembleSpec, build_ensemble, parse_spec

    cfg = _load(args)
    members = [m.strip() for m in args.members.split(",") if m.strip()] if args.members else []

    if args.spec:
        spec = parse_spec(args.spec, name=args.name, mode=args.mode, method=args.method)
    elif members:
        spec = EnsembleSpec(
            members=members,
            name=args.name or ("ensemble_" + "_".join(members)),
            mode=args.mode,
            method=args.method,
        )
    else:
        raise SystemExit("give either --members a,b,c or --spec 'a+b:2+c'")

    _sync_label_space(cfg, cfg.output_dir, spec.members[0])

    log.info(
        "members should be chosen before looking at their test scores; picking the "
        "best-performing subset after the fact is selection on the test set"
    )
    report = build_ensemble(cfg.output_dir, spec, overwrite=args.overwrite)

    from ppgl.experiments import results as store

    table = store.load_fold_metrics(cfg.output_dir, spec.name, mode="soft")
    print(f"\nensemble '{report.name}'  method={report.method}  members={len(report.members)}")
    print("  " + " + ".join(report.members))
    print(f"  folds built: {report.n_folds}")
    if report.skipped:
        print(f"  member-folds skipped (not in every member): {len(report.skipped)}")
    if not table.empty:
        for metric in ("patient_macro_f1", "macro_f1", "patient_mcc"):
            if metric in table.columns:
                values = table[metric].dropna()
                if len(values):
                    print(f"  {metric}: {values.mean():.4f} +/- {values.std(ddof=1):.4f}")
        print("\n  per-member reference:")
        for member in report.members:
            member_table = store.load_fold_metrics(cfg.output_dir, member, mode=spec.mode)
            if not member_table.empty and "patient_macro_f1" in member_table.columns:
                print(
                    f"    {member:<24} {member_table['patient_macro_f1'].mean():.4f}"
                )
    return 0


def cmd_selective(args) -> int:
    """Risk-coverage analysis from stored probabilities. No retraining."""
    from ppgl.experiments.selective import plot_risk_coverage, run_selective_analysis

    cfg = _load(args)
    experiments = (
        [e.strip() for e in args.experiments.split(",") if e.strip()]
        if args.experiments
        else _default_experiments(cfg.output_dir)
    )
    if not experiments:
        log.warning("no experiments found under %s", cfg.output_dir)
        return 1
    _sync_label_space(cfg, cfg.output_dir, experiments[0])

    out_dir = args.out_dir or os.path.join(cfg.output_dir, "selective")
    curves, summary = run_selective_analysis(
        cfg.output_dir,
        experiments,
        mode=args.mode,
        out_dir=out_dir,
    )
    if summary.empty:
        log.warning("no predictions available for selective analysis")
        return 1

    view = summary[(summary.level == args.level) & (summary.score == args.score)]
    columns = [
        "experiment",
        "full_coverage_error",
        "aurc",
        "eaurc",
        "failure_auroc",
        "error_at_coverage_50",
        "balanced_error_at_coverage_50",
        "classes_retained_at_coverage_50",
    ]
    print(f"\nselective prediction ({args.level} level, score={args.score})")
    print(view[columns].sort_values("aurc").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nfailure_auroc is the headline: 0.5 means confidence carries no "
        "information about whether the prediction is correct."
    )
    print(
        "Read balanced_error and classes_retained alongside the error column - a "
        "low error at low coverage is only meaningful if the rare clusters are "
        "still represented."
    )

    if not args.no_plot:
        path = plot_risk_coverage(
            curves,
            os.path.join(out_dir, f"risk_coverage_{args.level}_{args.score}.png"),
            level=args.level,
            score=args.score,
        )
        if path:
            print(f"\nwrote {path}")
    print(f"wrote {os.path.join(out_dir, 'selective_summary.csv')}")
    print(f"wrote {os.path.join(out_dir, 'risk_coverage_curves.csv')}")
    return 0


def cmd_locations(args) -> int:
    from ppgl.data.inventory import load_manifest
    from ppgl.data.location import build_heuristic_locations, load_location_csv

    cfg = _load(args)
    frame = load_manifest(cfg.data.manifest)

    if cfg.data.location_source == "csv" and cfg.data.location_csv:
        table = load_location_csv(cfg.data.location_csv, frame)
        print(table["region"].value_counts().to_string())
        return 0

    build_heuristic_locations(
        frame,
        output_csv=cfg.data.location_cache,
        review_csv=args.review_csv,
    )
    return 0


def cmd_folds(args) -> int:
    from ppgl.data.splits import fingerprint, save_fold_summary
    from ppgl.experiments.data_prep import build_folds, prepare_frame

    cfg = _load(args)
    frame = prepare_frame(cfg)
    folds = build_folds(cfg, frame)
    save_fold_summary(frame, folds, os.path.join(cfg.output_dir, "fold_summary.csv"))
    print(f"built {len(folds)} folds; fingerprint={fingerprint(folds)}")
    return 0


def cmd_train(args) -> int:
    from ppgl.models.names import is_tabular

    cfg = _load(args)
    setup_logging(level=args.log_level, job_tag=cfg.experiment)

    if is_tabular(cfg.model.name):
        from ppgl.experiments import run_tabular as runner
    else:
        from ppgl.experiments import run_cv as runner

    table = runner.run_cv(
        cfg,
        repeats=_split_ints(args.repeats),
        fold_ids=_split_ints(args.folds),
        device_index=args.device_index,
    )
    if table is not None and not table.empty:
        columns = [
            c
            for c in ("repeat", "fold", "macro_f1", "patient_macro_f1", "macro_auc_ovr", "mcc")
            if c in table.columns
        ]
        print(table[columns].to_string(index=False))
    return 0


def cmd_grid(args) -> int:
    from ppgl.experiments.dispatch import build_jobs, run_jobs

    setup_logging(level=args.log_level)
    paths = _expand_configs(args.configs)
    if not paths:
        log.error("no configs matched %s", args.configs)
        return 2

    configs = []
    for path in paths:
        cfg = load_config(path, overrides=args.set or [])
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.smoke_test:
            cfg.smoke_test = True
            cfg.optim.epochs = min(cfg.optim.epochs, 2)
            cfg.cv.n_repeats = 1
            cfg.cv.n_splits = min(cfg.cv.n_splits, 3)
            cfg.cv.require_all_classes_per_fold = False
        configs.append(cfg)
        log.info("queued experiment %r (model=%s)", cfg.experiment, cfg.model.name)

    jobs = build_jobs(configs, repeats=_split_ints(args.repeats), fold_ids=_split_ints(args.folds))
    records = run_jobs(jobs, n_parallel=args.n_parallel, log_level=args.log_level)

    failures = [r for r in records if r.get("status") == "failed"]
    return 1 if failures else 0


def cmd_analyse(args) -> int:
    from ppgl.experiments.routing_analysis import analyse

    cfg = _load(args)
    setup_logging(level=args.log_level)
    experiments = _split_list(args.experiments) or [cfg.experiment]
    for experiment in experiments:
        _sync_label_space(cfg, cfg.output_dir, experiment)
        analyse(cfg.output_dir, experiment)
    return 0


def cmd_stats(args) -> int:
    from ppgl.experiments.stats import run_all

    cfg = _load(args)
    setup_logging(level=args.log_level)
    _sync_label_space(cfg, cfg.output_dir, args.reference)
    others = _split_list(args.others) or [
        name for name in _default_experiments(cfg.output_dir) if name != args.reference
    ]
    run_all(
        cfg.output_dir,
        reference=args.reference,
        others=others,
        mode=args.mode,
        require_matching_folds=not args.allow_mismatched_folds,
    )
    return 0


def cmd_tables(args) -> int:
    from ppgl.experiments.tables import build_all

    cfg = _load(args)
    setup_logging(level=args.log_level)
    experiments = _split_list(args.experiments) or _default_experiments(cfg.output_dir)
    _sync_label_space(cfg, cfg.output_dir, args.reference)
    build_all(
        cfg.output_dir,
        experiments=experiments,
        reference=args.reference,
        manifest_path=cfg.data.manifest,
        mode=args.mode,
    )
    return 0


def cmd_figures(args) -> int:
    from ppgl.experiments.figures import build_all

    cfg = _load(args)
    setup_logging(level=args.log_level)
    experiments = _split_list(args.experiments) or _default_experiments(cfg.output_dir)
    _sync_label_space(cfg, cfg.output_dir, args.reference)
    build_all(
        cfg.output_dir,
        experiments=experiments,
        reference=args.reference,
        checkpoint=args.checkpoint,
    )
    return 0


def cmd_attention_comparison(args) -> int:
    """Side-by-side class-token attention for the control and the proposed model.

    Separate from ``figures`` because it is the only figure needing two checkpoints,
    and because it must be told which fold they came from. Everything else in the
    figure pipeline runs off the stored per-fold predictions alone.
    """
    from ppgl.experiments.figures import attention_comparison_figure

    cfg = _load(args)
    setup_logging(level=args.log_level)
    _sync_label_space(cfg, cfg.output_dir, args.proposed)
    written = attention_comparison_figure(
        cfg.output_dir,
        proposed=args.proposed,
        control=args.control,
        proposed_checkpoint=args.proposed_checkpoint,
        control_checkpoint=args.control_checkpoint,
        repeat=int(args.repeat),
        fold=int(args.fold),
        path=args.out,
    )
    if not written:
        log.error("attention comparison figure was not written; see the warnings above")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppgl",
        description="PPGL genetic-cluster classification from CT (ISBI 2027)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", help="YAML config file")
        sub.add_argument(
            "--set",
            nargs="*",
            default=[],
            metavar="KEY=VALUE",
            help="dotted config overrides, e.g. optim.lr=3e-5",
        )
        sub.add_argument("--output-dir", help="override output_dir")
        sub.add_argument("--experiment", help="override the experiment name")
        sub.add_argument("--log-level", default="INFO")
        sub.add_argument(
            "--smoke-test",
            action="store_true",
            help="tiny subset, few epochs: checks wiring, produces no valid results",
        )

    doctor = subparsers.add_parser("doctor", help="report environment and data status")
    add_common(doctor)
    doctor.set_defaults(func=cmd_doctor)

    fetch = subparsers.add_parser(
        "fetch-weights", help="download pretrained backbones into the local cache"
    )
    add_common(fetch)
    fetch.add_argument(
        "--backbones",
        help="comma-separated architecture names; default is the required set",
    )
    fetch.add_argument(
        "--all",
        action="store_true",
        help="also fetch the optional encoder-axis checkpoints (~430 MB more)",
    )
    fetch.add_argument("--force", action="store_true", help="re-download files already present")
    fetch.add_argument(
        "--verify-only",
        action="store_true",
        help="only check existing files against SHA256SUMS",
    )
    fetch.set_defaults(func=cmd_fetch_weights)

    inventory = subparsers.add_parser("inventory", help="build the dataset manifest")
    add_common(inventory)
    inventory.set_defaults(func=cmd_inventory)

    template = subparsers.add_parser(
        "clinical-template", help="write a blank clinical sheet for the chart review"
    )
    add_common(template)
    template.add_argument("--out", default="results/clinical_template.csv")
    template.set_defaults(func=cmd_clinical_template)

    locations = subparsers.add_parser("locations", help="build or inspect region labels")
    add_common(locations)
    locations.add_argument(
        "--review-csv",
        help="write heuristic labels sorted by ascending confidence for radiologist review",
    )
    locations.set_defaults(func=cmd_locations)

    folds = subparsers.add_parser("folds", help="build CV folds and write the audit summary")
    add_common(folds)
    folds.set_defaults(func=cmd_folds)

    train = subparsers.add_parser("train", help="run one experiment across folds")
    add_common(train)
    train.add_argument("--repeats", help="comma-separated repeat indices")
    train.add_argument("--folds", help="comma-separated fold indices")
    train.add_argument("--device-index", type=int, default=None, help="pin to a CUDA ordinal")
    train.set_defaults(func=cmd_train)

    grid = subparsers.add_parser("grid", help="run many experiments across all GPUs")
    add_common(grid)
    grid.add_argument("--configs", nargs="+", required=True, help="config paths or globs")
    grid.add_argument("--repeats", help="comma-separated repeat indices")
    grid.add_argument("--folds", help="comma-separated fold indices")
    grid.add_argument(
        "--n-parallel",
        type=int,
        default=0,
        help="worker processes; 0 = one per visible GPU (default)",
    )
    grid.set_defaults(func=cmd_grid)

    ensemble = subparsers.add_parser(
        "ensemble", help="combine stored predictions into a new pseudo-experiment (no GPU)"
    )
    add_common(ensemble)
    ensemble.add_argument("--members", help="comma-separated experiment names")
    ensemble.add_argument(
        "--spec", help="alternative syntax with optional weights, e.g. 'swin:2+vit:1+resnet101'"
    )
    ensemble.add_argument("--name", help="name for the ensemble experiment")
    ensemble.add_argument(
        "--method",
        default="mean",
        choices=["mean", "geometric", "val_weighted"],
        help="how member probabilities are combined (default: mean)",
    )
    ensemble.add_argument(
        "--mode", default="soft", help="which stored prediction mode to read from members"
    )
    ensemble.add_argument(
        "--overwrite", action="store_true", help="rebuild folds that already exist"
    )
    ensemble.set_defaults(func=cmd_ensemble)

    selective = subparsers.add_parser(
        "selective", help="risk-coverage / deferral analysis from stored probabilities (no GPU)"
    )
    add_common(selective)
    selective.add_argument("--experiments", help="comma-separated experiment names")
    selective.add_argument("--mode", default="soft")
    selective.add_argument(
        "--level", default="patient", choices=["slice", "patient"], help="reporting level"
    )
    selective.add_argument(
        "--score",
        default="msp",
        choices=["msp", "margin", "negent"],
        help="confidence score shown in the printed table (all are computed)",
    )
    selective.add_argument("--out-dir", help="where to write curves (default: <output_dir>/selective)")
    selective.add_argument("--no-plot", action="store_true", help="skip the PNG")
    selective.set_defaults(func=cmd_selective)

    analyse = subparsers.add_parser("analyse", help="routing analysis")
    add_common(analyse)
    analyse.add_argument("--experiments", help="comma-separated experiment names")
    analyse.set_defaults(func=cmd_analyse)

    stats = subparsers.add_parser("stats", help="paired comparisons with Holm correction")
    add_common(stats)
    stats.add_argument("--reference", required=True, help="the model being argued for")
    stats.add_argument("--others", help="comma-separated comparators (default: all)")
    stats.add_argument("--mode", default="soft")
    stats.add_argument(
        "--allow-mismatched-folds",
        action="store_true",
        help="permit comparing models evaluated on different splits (not recommended)",
    )
    stats.set_defaults(func=cmd_stats)

    tables = subparsers.add_parser("tables", help="build result tables (CSV + LaTeX)")
    add_common(tables)
    tables.add_argument("--experiments", help="comma-separated experiment names")
    tables.add_argument("--reference", default="routed_soft")
    tables.add_argument("--mode", default="soft")
    tables.set_defaults(func=cmd_tables)

    figures = subparsers.add_parser("figures", help="build figures")
    add_common(figures)
    figures.add_argument("--experiments", help="comma-separated experiment names")
    figures.add_argument("--reference", default="routed_soft")
    figures.add_argument("--checkpoint", help="checkpoint for Grad-CAM overlays")
    figures.set_defaults(func=cmd_figures)

    attn_cmp = subparsers.add_parser(
        "attention-comparison",
        help="control vs proposed class-token attention, one row per rare cluster",
    )
    add_common(attn_cmp)
    attn_cmp.add_argument("--proposed", default="amg", help="supervised experiment name")
    attn_cmp.add_argument("--control", default="amg_control", help="lambda=0 experiment name")
    attn_cmp.add_argument("--proposed-checkpoint", required=True, help="path to its model.pt")
    attn_cmp.add_argument("--control-checkpoint", required=True, help="path to its model.pt")
    attn_cmp.add_argument("--repeat", default=0, help="repeat index the checkpoints came from")
    attn_cmp.add_argument("--fold", default=0, help="fold index the checkpoints came from")
    attn_cmp.add_argument("--out", help="output path (default <output-dir>/figures/)")
    attn_cmp.set_defaults(func=cmd_attention_comparison)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(level=getattr(args, "log_level", "INFO"))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - surface a clean message, keep the traceback
        log.exception("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
