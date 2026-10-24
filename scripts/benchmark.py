"""T1 benchmark (Plan v3.1 section 7): seconds per step, peak memory, 2-concurrent-S throughput.

CPU/tiny scale validates the pipeline locally; --scale real on the Colab L4 produces the
numbers that re-cut the section 6 budget. Synthetic data throughout — no FineWeb needed.
Writes results/benchmark_<device>.md and .json.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dustq import qmodel  # noqa: E402
from dustq.config import SIZES  # noqa: E402
from dustq.dust_ext import DUSTQ  # noqa: E402
from dustq.recipes import recipe  # noqa: E402
from dustq.runner import TINY, build_model, tiny_batch  # noqa: E402
from dustq.upstream import ensure_upstream_on_path  # noqa: E402

SIZE_ALIASES = {"S": "L4d256", "M": "L8d512"}  # v3.1 cell notation -> SIZES keys

ensure_upstream_on_path()

from dust import make_optimizer as dust_make_optimizer  # noqa: E402


def make_job(size: str, population: int, precision: str, device, scale: str):
    kwargs = dict(TINY) if scale == "tiny" else dict(SIZES[size])
    model = build_model(kwargs, precision, device, seed=0).requires_grad_(False)
    settings = recipe("1M", population, model.config.n_layer)
    estimator = DUSTQ(model, settings, seed=0)
    if precision != "fp":
        _, bits, scheme = qmodel.parse_precision(precision)
        if scheme == "mse_alpha":
            scheme = "absmax"  # benchmark: timing is scale-scheme-independent
        # C5: benchmark timing is scale-scheme-independent, so low-bit absmax is allowed here
        qmodel.register_act_hooks(model, bits, scheme, allow_lowbit_absmax=True)
        qmodel.set_exact_path(model, enabled=True)  # measure the real (exact) path
    optimizer = dust_make_optimizer(model, settings)
    x, y = tiny_batch(device, seed=7, sequences=8, length=model.config.sequence_len,
                       vocab=model.config.vocab_size)
    return estimator, optimizer, x, y


def measure(size, population, precision, device, scale, steps):
    estimator, optimizer, x, y = make_job(size, population, precision, device, scale)
    estimator.step(x, y, optimizer)  # warmup
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        estimator.step(x, y, optimizer)
    per_step = (time.perf_counter() - t0) / steps
    peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else float("nan")
    estimator.close()
    return per_step, peak


def concurrent_test(device, scale, jobs: int = 2, steps: int = 2):
    """M3: 2-concurrent-S via subprocesses — threads in one process share a single CUDA
    stream and measure nothing; independent processes interleave kernels on the GPU."""
    cmd = [sys.executable, __file__, "--worker", "--device", device.type, "--worker-scale", scale,
           "--worker-size", "L4d256", "--worker-population", "256", "--worker-precision", "fp",
           "--worker-steps", str(steps)]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    procs = [subprocess.Popen(cmd) for _ in range(jobs)]
    for p in procs:
        p.wait()
    wall = time.perf_counter() - t0
    if any(p.returncode for p in procs):
        raise RuntimeError(f"a concurrency worker failed (returncodes {[p.returncode for p in procs]})")
    return wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--scale", default="tiny", choices=["tiny", "real"])
    parser.add_argument("--sizes", default="S,M", help="S/M aliases or SIZES keys (L4d256, L8d512)")
    parser.add_argument("--populations", default="256,1024,4096")
    parser.add_argument("--precisions", default="fp,w1.58a4")
    parser.add_argument("--steps", type=int, default=3, help="measured steps per cell")
    parser.add_argument("--skip-concurrent", action="store_true", help="skip the 2-worker test (fast paths)")
    parser.add_argument("--mem16-probe", action="store_true",
                        help="also measure the nrep=16 (K=16384) memory ceiling on M (~26 min)")
    # worker mode (spawned by concurrent_test; also usable by hand)
    parser.add_argument("--worker", action="store_true", help="run ONE job and print JSON timing")
    parser.add_argument("--worker-size", default="L4d256")
    parser.add_argument("--worker-population", default="256")
    parser.add_argument("--worker-precision", default="fp")
    parser.add_argument("--worker-steps", type=int, default=2)
    parser.add_argument("--worker-scale", default="tiny",
                        help="tiny|real; concurrent_test always passes this explicitly")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.worker:
        per_step, _ = measure(args.worker_size, int(args.worker_population),
                              args.worker_precision, device, args.worker_scale,
                              max(args.worker_steps, 1))
        print(json.dumps({"s_per_step": round(per_step, 3)}))
        return

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    rows = []
    populations = [int(k) for k in args.populations.split(",")]
    if args.scale == "tiny":
        cells = [("tiny", k, p) for k in populations for p in args.precisions.split(",")]
    else:
        resolved = [SIZE_ALIASES.get(s.strip(), s.strip()) for s in args.sizes.split(",")]
        for s in resolved:
            if s not in SIZES:
                raise SystemExit(f"unknown size {s!r}; use S/M or one of {sorted(SIZES)}")
        cells = [(s, k, p) for s in resolved for k in populations
                 for p in args.precisions.split(",")]
    for size, population, precision in cells:
        try:
            s_per_step, peak = measure(size, population, precision, device, args.scale, args.steps)
            rows.append(dict(size=size, K=population, precision=precision,
                             s_per_step=round(s_per_step, 3),
                             peak_gb=round(peak, 2) if peak == peak else ""))
            print(f"{size} K={population} {precision}: {s_per_step:.3f} s/step, peak {peak:.2f} GB", flush=True)
        except RuntimeError as exc:  # e.g. a CUDA OOM ceiling on one cell: record, continue
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rows.append(dict(size=size, K=population, precision=precision, s_per_step="",
                             peak_gb="", note=f"error: {str(exc)[:120]}"))
            print(f"{size} K={population} {precision}: FAILED — {str(exc)[:120]}", flush=True)

    # nrep=16 memory probe (K=16384) at real scale, opt-in (--mem16-probe): ~26 min on
    # its own; the matrix tops at nrep=8 and the extrapolation is already recorded.
    mem16 = None
    if args.scale == "real" and args.mem16_probe:
        probe_size = SIZE_ALIASES.get("M", "M")  # aliases must resolve here too (B2-class bug)
        try:
            _, peak = measure(probe_size, 16384, "fp", device, args.scale, 1)
            mem16 = round(peak, 2)
            rows.append(dict(size=probe_size, K=16384, precision="fp", s_per_step="", peak_gb=mem16,
                             note="nrep=16 memory probe (1 step)"))
        except RuntimeError as exc:
            rows.append(dict(size=probe_size, K=16384, precision="fp", s_per_step="", peak_gb="",
                             note=f"nrep=16 probe OOM/error: {str(exc)[:200]}"))

    conc = None
    if args.scale == "real" and not args.skip_concurrent:
        try:
            conc = concurrent_test(device, args.scale)
        except Exception as exc:
            conc = None
            rows.append(dict(size="S(2x)", K=256, precision="fp", s_per_step="",
                             peak_gb="", note=f"concurrent test failed: {str(exc)[:120]}"))

    out = Path("results")
    out.mkdir(exist_ok=True)
    lines = [f"# T1 benchmark — {gpu} — scale={args.scale} — {time.strftime('%Y-%m-%d %H:%M')}", "",
             "| size | K | precision | s/step | peak GB |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['size']} | {r['K']} | {r['precision']} | {r['s_per_step']} | {r['peak_gb']} |"
                     + (f" ({r['note']})" if r.get("note") else ""))
    if conc:
        lines += ["", f"2-concurrent S/K256 (2 steps each): {conc:.1f} s wall vs single-run "
                      f"{[r['s_per_step'] for r in rows if r['size'] == 'S' and r['K'] == 256][0]} s/step."]
    (out / f"benchmark_{device.type}.md").write_text("\n".join(lines) + "\n")
    (out / f"benchmark_{device.type}.json").write_text(json.dumps(
        dict(gpu=gpu, scale=args.scale, rows=rows, concurrent_wall_s=conc), indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
