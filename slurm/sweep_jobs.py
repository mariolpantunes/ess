"""Write the job list of the fixes sweep: one line per cell, `RUN_ID ARM ARGS...`.

Arms are source trees beside the submit folder (see `slurm/sweep.sbatch`):
`head` (this commit), `batch` (41cf3b7, placement before the sequential
fix), `l1cap` (52456fa, second L1 step cap below p = 1) and `random`
(uniform points, the scale).

Run: python slurm/sweep_jobs.py > slurm/sweep_jobs.txt
"""

import random

DIMS = (2, 3, 5, 8, 16, 32)
SIZES = (256, 1024)
RATIOS = (0.5, 1.0, 5.0)

# name, arm, extra args, needs existing points (a placement question)
P1 = [
    ("head_c005", "head", "--cap 0.005", False),
    ("head_c01", "head", "--cap 0.01", False),
    ("head_c02", "head", "--cap 0.02", False),
    ("head_c04", "head", "--cap 0.04", False),
    ("random", "random", "", False),
    ("head_pool1", "head", "--cap 0.01 --pool 1", True),
    ("head_pool16", "head", "--cap 0.01 --pool 16", True),
    ("head_pool256", "head", "--cap 0.01 --pool 256", True),
    ("head_place", "head", "--epochs 0", True),
    ("batch_c02", "batch", "--cap 0.02", True),
    ("batch_c01", "batch", "--cap 0.01", True),
    ("batch_place", "batch", "--epochs 0", True),
]


def lp(tag: str, p: float) -> list:
    return [
        (f"head_{tag}_c01", "head", f"--p {p} --cap 0.01", False),
        (f"head_{tag}_c02", "head", f"--p {p} --cap 0.02", False),
        (f"l1cap_{tag}", "l1cap", f"--p {p} --cap 0.02", False),
    ]


def main() -> None:
    lines = []
    for n0 in SIZES:
        variants = P1 + lp("p05", 0.5) + (lp("p025", 0.25) if n0 == 256 else [])
        for d in DIMS:
            cells = [("scratch", 1.0)] + [(s, r) for s in ("uniform", "clustered") for r in RATIOS]
            for scenario, ratio in cells:
                for name, arm, extra, placement in variants:
                    if placement and scenario == "scratch":
                        continue
                    run = f"{scenario}_d{d}_n{n0}_r{ratio:g}_{name}"
                    lines.append(
                        f"{run} {arm} --scenario {scenario} --d {d} --n0 {n0} --ratio {ratio:g} {extra}".strip())
    random.Random(0).shuffle(lines)  # mix sizes across shards
    print("\n".join(lines))


if __name__ == "__main__":
    main()
