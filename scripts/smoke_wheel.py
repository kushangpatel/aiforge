"""Smoke-test an installed AI Forge wheel.

Run it with the interpreter of a fresh virtualenv the wheel was installed into,
from a directory outside the repository (and with ``-I`` so the checkout is not
on ``sys.path``)::

    /tmp/venv/bin/python -I scripts/smoke_wheel.py

It checks that ``aiforge`` imports from site-packages, that its version matches
the installed distribution's metadata, that the runtime data files shipped, and
that the ``aiforge`` console script runs ``--help`` and ``init``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    import aiforge

    package_dir = Path(aiforge.__file__).resolve().parent
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    if site_packages not in package_dir.parents:
        _fail(f"aiforge imported from {package_dir}, not from {site_packages}")

    dist_version = importlib.metadata.version("aiforge-kit")
    if aiforge.__version__ != dist_version:
        _fail(f"aiforge.__version__ {aiforge.__version__} != metadata {dist_version}")

    templates = importlib.resources.files("aiforge.templates")
    data_files = [
        package_dir / "py.typed",
        Path(str(templates.joinpath(".env.example"))),
        Path(str(templates.joinpath("aiforge.yaml.example"))),
        package_dir / "playground" / "app.py",
        package_dir / "playground" / "pages" / "settings.py",
    ]
    missing = [str(p) for p in data_files if not p.is_file()]
    if missing:
        _fail(f"missing package data: {missing}")

    cli = Path(sys.executable).parent / ("aiforge.exe" if os.name == "nt" else "aiforge")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with tempfile.TemporaryDirectory() as tmp:
        for args in (["--help"], ["init"]):
            proc = subprocess.run(
                [str(cli), *args], cwd=tmp, env=env, capture_output=True, text=True
            )
            if proc.returncode != 0:
                _fail(f"`aiforge {' '.join(args)}` exited {proc.returncode}:\n{proc.stderr}")
        created = sorted(p.name for p in Path(tmp).iterdir())
        if "aiforge.yaml" not in created:
            _fail(f"`aiforge init` did not create aiforge.yaml (created: {created})")

    print(f"OK: aiforge {dist_version} from {package_dir}")


if __name__ == "__main__":
    main()
