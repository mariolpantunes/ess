"""Every example still runs.

`examples/` is not imported by `src`, is not on the lint gate's path and had
no tests, so it rots silently and the rot is invisible: when `ess.utils`
dropped its Clark-Evans index, `benchmark_dispersion.py` and
`benchmark_reexploration.py` both kept asking for the removed key and died in
their *reporting* — after completing every run. `evaluate_esa.py` was worse,
because it did not crash at all: it kept printing a discrepancy under a
Clark-Evans heading, which inverted the reading and made a 10x improvement
look like a 10x regression for weeks.

Importing is not enough to catch that class. These are dict lookups inside a
report function, reached only when the script actually runs to the end. So
each example is executed as a subprocess with the smallest arguments it
accepts, and the test asserts it exits 0. That is why several of them grew a
`--dims` or an equivalent knob: a benchmark you cannot run small is a
benchmark nothing can check.

Plot scripts that block on `plt.show()` run under a headless backend.
"""

import importlib
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# The smallest arguments each script accepts. A script absent from this table
# fails the test rather than being skipped: adding an example should mean
# deciding how it gets checked, not quietly opting out.
SMALLEST = {
    "smoke_esa": ["--dimensions", "2"],
    "benchmark_dispersion": ["--phase", "force", "--dims", "2",
                             "--tune-seeds", "1"],
    "benchmark_reexploration": ["--dims", "2", "--anchors", "20",
                                "--new", "10", "--seeds", "1"],
    "profile_esa": ["--n", "100"],
    "profile_torann": ["--sizes", "200", "--dims", "4", "--epochs", "1"],
    "evaluate_esa": [],          # no CLI; it is small enough as it stands
    "sample_2d_00": [],
    "sample_2d_01": [],
    "sample_3d_00": [],
    "sample_3d_01": [],
}

# Import-time optional dependencies. A script whose dependency is missing
# skips rather than fails: this suite must stay green on a bare checkout, and
# CI installs the `dev` extra precisely so these actually run there.
OPTIONAL = {
    "sample_2d_00": ("matplotlib",),
    "sample_2d_01": ("matplotlib",),
    "sample_3d_00": ("matplotlib",),
    "sample_3d_01": ("matplotlib",),
}

TIMEOUT = 600


def _missing(name):
    """Optional dependencies of `name` that this environment lacks."""
    out = []
    for dep in OPTIONAL.get(name, ()):
        try:
            importlib.import_module(dep)
        except ImportError:
            out.append(dep)
    return out


class TestExamplesRun(unittest.TestCase):
    """One subtest per script, so one breakage does not mask the rest."""

    def test_every_example_runs(self):
        scripts = sorted(p.stem for p in EXAMPLES.glob("*.py")
                         if not p.stem.startswith("_"))
        self.assertTrue(scripts, "no examples found — is the layout still right?")
        self.assertEqual(
            set(scripts) - set(SMALLEST), set(),
            "example added without deciding how the smoke test runs it; "
            "add it to SMALLEST")

        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"          # plt.show() must not block
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "src"), env.get("PYTHONPATH", "")])

        for name in scripts:
            with self.subTest(script=name):
                missing = _missing(name)
                if missing:
                    self.skipTest(f"needs {', '.join(missing)}")
                with tempfile.TemporaryDirectory() as tmp:
                    # --out lands here rather than in the working tree; the
                    # scripts that write into examples/out are left alone,
                    # since overwriting a local artifact is not a failure.
                    args = list(SMALLEST[name])
                    if name == "smoke_esa":
                        args += ["--out", os.path.join(tmp, "out.json")]
                    proc = subprocess.run(
                        [sys.executable, str(EXAMPLES / f"{name}.py"), *args],
                        cwd=ROOT, env=env, capture_output=True, text=True,
                        timeout=TIMEOUT, check=False,
                    )
                self.assertEqual(
                    proc.returncode, 0,
                    f"{name}.py exited {proc.returncode}\n"
                    f"--- stderr ---\n{proc.stderr[-3000:]}")


if __name__ == "__main__":
    unittest.main()
