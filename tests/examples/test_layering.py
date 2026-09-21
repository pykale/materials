"""Keep run-configuration handling in examples/."""

import ast

from ..conftest import LIBRARY_ROOT

FORBIDDEN = ("config", "runner", "case_study_references", "hydra", "omegaconf")


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_library_imports():
    stages = {child.name for child in LIBRARY_ROOT.iterdir() if child.is_dir() and not child.name.startswith("_")}
    modules = sorted(LIBRARY_ROOT.rglob("*.py"))
    assert len(modules) > 20
    for path in modules:
        for name in imports(path):
            root = name.split(".")[0]
            assert root not in FORBIDDEN, f"{path.name} imports {name}"
            assert root not in stages, f"{path.name} uses a bare stage import {name}; use kalematerials.{name}"


def test_evaluate_never_fits():
    for path in (LIBRARY_ROOT / "evaluate").glob("*.py"):
        assert not any(marker in path.read_text() for marker in (".fit(", ".train(", "fit_models")), path.name
