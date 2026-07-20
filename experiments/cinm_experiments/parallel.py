"""Generic parallel-execution helpers shared across experiments: a plain
worker-pool map, and a resource-capped scheduler for hardware runs that share
a fixed budget (e.g. total DPUs in use across concurrently running
benchmarks)."""
from __future__ import annotations

import concurrent.futures as cf
from typing import Callable, Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(items: Iterable[T], fn: Callable[[T], R], *, workers: int,
                  desc: str = "", use_threads: bool = False) -> list[R]:
    """Run fn(item) for every item with a worker pool and a progress bar.

    Use use_threads=True for subprocess-heavy / I/O-bound work (the default,
    ProcessPoolExecutor, is for CPU-bound work and requires fn and items to be
    picklable -- i.e. module-level functions, not closures)."""
    items = list(items)
    executor_cls = cf.ThreadPoolExecutor if use_threads else cf.ProcessPoolExecutor
    results = []
    with executor_cls(max_workers=workers) as ex:
        futures = [ex.submit(fn, item) for item in items]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc=desc):
            results.append(fut.result())
    return results


def run_resource_capped(items: Iterable[T], fn: Callable[[T], R], *,
                         cost_fn: Callable[[T], int], cap: int, workers: int,
                         desc: str = "",
                         should_retry: Callable[[R], bool] = lambda r: False) -> list[R]:
    """Run fn(item) in a thread pool, never letting the sum of cost_fn(item)
    over in-flight items exceed `cap` -- e.g. a DPU budget shared across
    concurrently running hardware benchmarks. If a single item's cost exceeds
    the cap it still runs, alone. Results for which should_retry(result) is
    true are retried once, sequentially, after everything else has drained
    (for transient allocation races)."""
    pending = list(items)
    total = len(pending)
    in_flight: dict[cf.Future, tuple[T, int]] = {}
    used = 0
    retry_pending: list[T] = []
    results: list[R] = []

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        pbar = tqdm(total=total, desc=desc)
        while pending or in_flight:
            remaining = []
            for item in pending:
                c = cost_fn(item)
                if used + c <= cap:
                    fut = ex.submit(fn, item)
                    in_flight[fut] = (item, c)
                    used += c
                else:
                    remaining.append(item)
            pending = remaining

            if not in_flight and pending:
                item = pending.pop(0)
                c = cost_fn(item)
                fut = ex.submit(fn, item)
                in_flight[fut] = (item, c)
                used += c

            done, _ = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                item, c = in_flight.pop(fut)
                used -= c
                result = fut.result()
                if should_retry(result):
                    retry_pending.append(item)
                else:
                    results.append(result)
                    pbar.update(1)

        for item in retry_pending:
            results.append(fn(item))
            pbar.update(1)
        pbar.close()
    return results
