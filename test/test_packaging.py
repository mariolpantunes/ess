"""Tests that `requirements.txt` and `pyproject.toml` agree.

They have drifted before -- the numpy floor read 2.4.0 in one file and 2.0.0
in the other -- and nothing breaks when they do, so nothing surfaces it. This
is the assertion that does.

`pyproject.toml` is read with a small regex rather than `tomllib`, which is
3.11+ while this package supports 3.10. Only the `dependencies` array is
needed, and it is a flat list of quoted strings.
"""

import os
import re
import unittest

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def _name_of(spec):
    """`"numpy>=2.0.0"` -> `"numpy"`."""
    return re.split(r"[<>=~!\[ ]", spec, maxsplit=1)[0].strip().lower()


def _dependencies():
    """The `dependencies` array of `[project]`, as written."""
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as handle:
        body = handle.read()
    match = re.search(r"^dependencies\s*=\s*\[(.*?)\]", body, re.S | re.M)
    if match is None:
        return []
    return re.findall(r"[\"']([^\"']+)[\"']", match.group(1))


def _requirements():
    """`name -> full specifier`, skipping comments and blanks."""
    out = {}
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                out[_name_of(line)] = line
    return out


class TestPackaging(unittest.TestCase):
    def setUp(self):
        self.dependencies = _dependencies()
        self.declared = _requirements()

    def test_the_dependency_list_was_actually_found(self):
        """A regex that silently matched nothing would pass every test below."""
        self.assertTrue(self.dependencies)

    def test_requirements_covers_every_runtime_dependency(self):
        missing = {_name_of(d) for d in self.dependencies} - set(self.declared)
        self.assertFalse(
            missing, f"in pyproject.toml but not requirements.txt: {sorted(missing)}")

    def test_runtime_floors_agree_between_the_two_files(self):
        for spec in self.dependencies:
            with self.subTest(dependency=spec):
                self.assertEqual(spec, self.declared[_name_of(spec)])
