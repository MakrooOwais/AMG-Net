"""Job dispatcher: runs the experiment grid across however many GPUs exist.

The grid is roughly ``n_models x n_repeats x n_splits`` short training jobs -
around 400 for the default configuration, each on ~570 slices for up to 60
epochs. At that size the right unit of parallelism is the *fold*, not the
minibatch: DDP inside a single fold spends more wall-clock time synchronising
gradients than computing them, whereas independent folds scale linearly and need
no inter-process communication at all.

So the dispatcher:

* discovers the visible GPU count at runtime (0, 1, or N),
* starts one worker process per GPU and pins it with ``CUDA_VISIBLE_DEVICES``,
* pulls ``(experiment, repeat, fold)`` jobs off a shared queue,
* skips jobs whose results already exist, so an interrupted grid resumes.

On a CPU- or MPS-only machine it degrades to a single sequential worker.
DDP remains available for one-off single-configuration runs via
``runtime.use_ddp`` and ``torchrun``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_module
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# How long a worker blocks on the job queue before re-checking. Only affects how
# quickly a worker notices a sentinel, never correctness.
_QUEUE_TIMEOUT_S = 2.0

from ppgl.utils.device import gpu_count, resolve_parallel_jobs
from ppgl.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


@dataclass
class Job:
    """One unit of work: a config plus the fold it should run."""

    experiment: str
    config_overrides: Dict[str, Any]
    repeat: int
    fold: int
    is_tabular: bool = False

    @property
    def tag(self) -> str:
        return f"{self.experiment}/r{self.repeat}f{self.fold}"


def build_jobs(configs: Sequence[Any], repeats: Optional[Sequence[int]] = None,
               fold_ids: Optional[Sequence[int]] = None) -> List[Job]:
    """Expand configs into per-fold jobs.

    Ordering matters for throughput: the cheap tabular controls are queued first
    so that the location-only result - the experiment that can invalidate the
    paper's framing - is available within minutes rather than after the whole
    deep grid.
    """
    from ppgl.models.names import is_tabular

    jobs: List[Job] = []
    for cfg in sorted(configs, key=lambda c: (not is_tabular(c.model.name), c.experiment)):
        selected_repeats = list(repeats) if repeats is not None else list(range(int(cfg.cv.n_repeats)))
        selected_folds = list(fold_ids) if fold_ids is not None else list(range(int(cfg.cv.n_splits)))
        for repeat in selected_repeats:
            for fold in selected_folds:
                jobs.append(
                    Job(
                        experiment=cfg.experiment,
                        config_overrides=cfg.to_dict(),
                        repeat=repeat,
                        fold=fold,
                        is_tabular=is_tabular(cfg.model.name),
                    )
                )
    return jobs


def _rebuild_config(payload: Dict[str, Any]):
    """Reconstruct a Config inside a worker process from its dict form.

    ``_validate`` is re-run here deliberately. ``load_config`` validates in the parent,
    but the worker receives a dict that has crossed a process boundary, and a
    round-trip that is not value-preserving hands the worker a config the parent would
    have rejected. That happened: ``data.location_source="none"`` was coerced back to
    ``None`` on the way in, and because nothing revalidated, the failure surfaced 500
    times from deep inside the data layer instead of once, here, naming the field.

    ``_validate`` is pure - it checks invariants and normalises two enum spellings - so
    running it a second time costs nothing and cannot mask a parent-side error.
    """
    from ppgl.config import _from_dict, _validate  # internal: round-trips to_dict()

    cfg = _from_dict(payload)
    _validate(cfg)
    return cfg


def _run_single_job(job: Job, device_index: Optional[int]) -> Dict[str, Any]:
    """Execute one job. Returns a status record; never raises into the pool."""
    from ppgl.experiments import results as store

    cfg = None
    try:
        cfg = _rebuild_config(job.config_overrides)

        if cfg.runtime.resume and store.is_fold_complete(
            cfg.output_dir,
            cfg.experiment,
            job.repeat,
            job.fold,
            expected_fingerprint=store.config_fingerprint(cfg),
        ):
            return {"job": job.tag, "status": "skipped"}

        if job.is_tabular:
            from ppgl.experiments import run_tabular

            run_tabular.run_cv(cfg, repeats=[job.repeat], fold_ids=[job.fold])
        else:
            from ppgl.experiments import run_cv as cv_module

            cv_module.run_cv(
                cfg,
                repeats=[job.repeat],
                fold_ids=[job.fold],
                device_index=device_index,
            )
        return {"job": job.tag, "status": "ok"}
    except Exception as exc:  # noqa: BLE001 - a failed fold must not kill the grid
        formatted = traceback.format_exc()
        # Persist the traceback next to the fold's results. Without this, the grid
        # only reports a one-line summary and diagnosing a failure means
        # re-running a fold by hand - unaffordable when 550 jobs share one cause.
        written_to = _write_failure(cfg, job, formatted)
        return {
            "job": job.tag,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": formatted,
            "traceback_file": written_to,
        }


def _write_failure(cfg: Optional[Any], job: Job, formatted: str) -> Optional[str]:
    """Write a failure traceback to the fold directory. Best effort."""
    from ppgl.experiments import results as store

    try:
        if cfg is not None:
            directory = store.fold_dir(cfg.output_dir, cfg.experiment, job.repeat, job.fold)
        else:
            output_dir = (job.config_overrides or {}).get("output_dir", "results")
            directory = store.fold_dir(output_dir, job.experiment, job.repeat, job.fold)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "failure.txt")
        with open(path, "w") as handle:
            handle.write(f"job: {job.tag}\n\n{formatted}")
        return path
    except Exception:  # noqa: BLE001 - never let diagnostics mask the real error
        return None


def _limit_worker_threads(n_threads: int = 1) -> None:
    """Stop each worker from trying to use the whole machine.

    Parallelism here is at the fold level: N workers each own one GPU. Every
    numeric library underneath, though, defaults to spawning one thread per
    *core*, so on a 96-core host with 8 workers the process tree asks for ~768
    threads. For the tabular models that is catastrophic rather than merely
    wasteful, because the data is 630 rows x 57 features - there is almost no
    work per thread, so the cost is entirely thread-barrier synchronisation.

    Measured on the real feature table (630x57, 4 classes, 400 boosting
    iterations): 0.75 s with one thread, 3.94 s with 14 threads on a 14-core
    machine. On the 96-core server with 8 concurrent workers the same fit was
    taking a median of 27 minutes - roughly a 2000x penalty - which stalled the
    grid on models that should be the cheapest in it.

    Must run before numpy/sklearn/lightgbm/torch are imported in this process:
    OpenMP reads these variables once, at library initialisation. Workers are
    started with the "spawn" method, so at this point the child is a fresh
    interpreter and nothing numeric has loaded yet - which is exactly why this
    belongs at the top of the worker entry point rather than in run_jobs.
    """
    value = str(int(n_threads))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        # Respect an explicit setting from the caller's environment.
        os.environ.setdefault(variable, value)

    # Only touch torch if this process has already loaded it. Importing it here
    # would cost seconds in the tabular workers that never need it, and under
    # spawn the environment variables above are read at import time anyway.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.set_num_threads(int(n_threads))
        except Exception:  # noqa: BLE001 - already fixed by an earlier call
            pass


def _worker(
    worker_id: int,
    device_id: Optional[int],
    queue: "mp.Queue",
    result_queue: "mp.Queue",
    log_level: str,
) -> None:
    """Worker loop. Pinned to one GPU for its whole lifetime."""
    if device_id is not None:
        # Pin before torch initialises CUDA, so the process sees exactly one
        # device and internally addresses it as index 0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    _limit_worker_threads()
    setup_logging(level=log_level, job_tag=f"w{worker_id}", force=True)
    worker_log = get_logger(f"ppgl.worker{worker_id}")
    worker_log.info("worker %d started (device_id=%s)", worker_id, device_id)

    # Shutdown is driven by an explicit ``None`` sentinel, one per worker, and the
    # get is blocking. An earlier version used get_nowait() and treated any
    # exception as "queue exhausted", which is wrong: multiprocessing.Queue feeds
    # its pipe from a background thread, so get_nowait() raises Empty while items
    # are still in flight. Workers that lost that race exited after their first
    # job, leaving most GPUs idle for the rest of the grid.
    while True:
        try:
            job = queue.get(timeout=_QUEUE_TIMEOUT_S)
        except queue_module.Empty:
            # No sentinel yet and nothing available: the producer may still be
            # feeding. Keep waiting rather than exiting.
            continue
        except (EOFError, OSError) as exc:
            worker_log.warning("worker %d queue closed (%s); exiting", worker_id, exc)
            break

        if job is None:  # sentinel: this worker is done
            break

        # device_index 0: CUDA_VISIBLE_DEVICES has already remapped the ordinal,
        # so the worker addresses its pinned GPU as 0 while the log records the
        # physical device.
        record = _run_single_job(job, device_index=0 if device_id is not None else None)
        status = record["status"]
        if status == "failed":
            worker_log.error("job %s FAILED: %s", job.tag, record.get("error"))
            # The last few frames are usually enough to identify the cause without
            # opening the file.
            tail = [line for line in (record.get("traceback") or "").splitlines() if line.strip()]
            for line in tail[-6:]:
                worker_log.error("    %s", line)
            if record.get("traceback_file"):
                worker_log.error("    full traceback: %s", record["traceback_file"])
        elif status == "skipped":
            worker_log.info("job %s already complete, skipped", job.tag)
        else:
            worker_log.info("job %s done (gpu=%s)", job.tag, device_id)
        result_queue.put(record)

    worker_log.info("worker %d finished", worker_id)


def run_jobs(
    jobs: Sequence[Job],
    n_parallel: int = 0,
    log_level: str = "INFO",
) -> List[Dict[str, Any]]:
    """Run all jobs, in parallel across GPUs where available.

    Returns one status record per job. A failing fold is recorded and the grid
    continues, so one bad configuration does not discard hours of work.
    """
    jobs = list(jobs)
    if not jobs:
        log.warning("no jobs to run")
        return []

    n_gpu = gpu_count()
    workers = resolve_parallel_jobs(n_parallel)

    if workers <= 1:
        log.info(
            "running %d jobs sequentially (gpus=%d)", len(jobs), n_gpu
        )
        # Best-effort here rather than guaranteed: this runs in the parent, so a
        # library already imported has fixed its thread pool. It still works for
        # the ones that matter, because sklearn and lightgbm are imported lazily
        # at fit time. LightGBM is additionally pinned via n_jobs at construction.
        _limit_worker_threads()
        records = []
        for job in jobs:
            log.info("running %s", job.tag)
            record = _run_single_job(job, device_index=0 if n_gpu > 0 else None)
            if record["status"] == "failed":
                log.error("job %s failed: %s", job.tag, record.get("error"))
            records.append(record)
        _log_summary(records)
        return records

    log.info("running %d jobs across %d workers (gpus=%d)", len(jobs), workers, n_gpu)

    # "spawn" is required: CUDA cannot be re-initialised in a forked child.
    context = mp.get_context("spawn")
    job_queue: "mp.Queue" = context.Queue()
    result_queue: "mp.Queue" = context.Queue()
    for job in jobs:
        job_queue.put(job)
    # One sentinel per worker, so each exits exactly once and only after the real
    # jobs are gone.
    for _ in range(workers):
        job_queue.put(None)

    processes = []
    for worker_id in range(workers):
        device_id = worker_id % n_gpu if n_gpu > 0 else None
        process = context.Process(
            target=_worker,
            args=(worker_id, device_id, job_queue, result_queue, log_level),
            daemon=False,
        )
        process.start()
        processes.append(process)

    records: List[Dict[str, Any]] = []
    finished = 0
    while finished < len(jobs):
        try:
            record = result_queue.get(timeout=5.0)
            records.append(record)
            finished += 1
            if finished % 10 == 0 or finished == len(jobs):
                log.info("progress: %d/%d jobs reported", finished, len(jobs))
            continue
        except queue_module.Empty:
            pass

        # Only conclude that work has stopped after a quiet period with every
        # worker dead, so a slow job is never mistaken for a crashed grid.
        if not any(process.is_alive() for process in processes):
            log.error(
                "all workers exited with %d/%d jobs reported; %d jobs did not run "
                "(check the per-worker logs above)",
                finished,
                len(jobs),
                len(jobs) - finished,
            )
            break

    for process in processes:
        process.join(timeout=120)
        if process.is_alive():
            log.warning("worker pid=%s did not exit; terminating", process.pid)
            process.terminate()

    _log_summary(records)
    return records


def _log_summary(records: Sequence[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    log.info("job summary: %s", counts)
    failures = [record for record in records if record["status"] == "failed"]
    if not failures:
        return

    # Group by error text: a grid failure almost always has one root cause, and
    # 550 identical lines obscures that rather than showing it.
    by_error: Dict[str, List[str]] = {}
    for failure in failures:
        by_error.setdefault(str(failure.get("error")), []).append(failure["job"])
    log.error("%d job(s) failed, %d distinct cause(s):", len(failures), len(by_error))
    for error, jobs in sorted(by_error.items(), key=lambda item: -len(item[1])):
        sample = ", ".join(jobs[:3]) + (f", +{len(jobs) - 3} more" if len(jobs) > 3 else "")
        log.error("  [%d jobs] %s", len(jobs), error)
        log.error("           affected: %s", sample)
    first = failures[0]
    if first.get("traceback_file"):
        log.error("full traceback for %s: %s", first["job"], first["traceback_file"])


__all__ = ["Job", "build_jobs", "run_jobs"]
