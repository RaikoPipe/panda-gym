"""run_meta.json: provenance of every training and evaluation run (git state, arguments, config, versions)."""
import dataclasses
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ("panda_gym", "stable_baselines3", "sb3_contrib", "torch", "pybullet", "gymnasium", "numpy")
REPO = Path(__file__).resolve().parent.parent


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(x):
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    return str(x)


class RunMeta:
    """Writes run_meta.json at start (end=None) and again at finish()."""

    def __init__(self, path, args, config=None, **extra):
        self.path = Path(path)
        self.meta = {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "argv": sys.argv,
            "args": vars(args) if hasattr(args, "__dict__") else args,
            "config": config,
            "versions": {p: _version(p) for p in PACKAGES},
            "host": socket.gethostname(),
            "start": now(),
            "end": None,
            **extra,
        }
        self.write()

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.meta, indent=2, default=_jsonable) + "\n")

    def finish(self, **extra):
        self.meta.update(end=now(), **extra)
        self.write()
