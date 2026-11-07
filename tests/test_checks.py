"""The script checks in tests/checks (each exits non-zero on a failure), on a GPU. The Q38 ones need ttf.q38 (ttf-core
with the Q38 package) and transformers >= 5.19 (Qwen3.5-MoE)."""
import glob
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


def _params():
    for p in sorted(glob.glob(os.path.join(ROOT, "tests", "checks", "*.py"))):
        n = os.path.basename(p)[:-3]
        marks = [pytest.mark.gpu]
        if n.startswith("q38_") and not (_has("ttf.q38") and _has("transformers.models.qwen3_5_moe")):
            marks.append(pytest.mark.skip(reason="needs ttf.q38 and transformers >= 5.19"))
        yield pytest.param(p, id=n, marks=marks)


@pytest.mark.parametrize("script", _params())
def test_check(script):
    if not __import__("torch").cuda.is_available():
        pytest.skip("no CUDA GPU")
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
