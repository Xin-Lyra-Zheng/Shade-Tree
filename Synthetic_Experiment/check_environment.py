"""Report whether the prespecified experiment dependencies are available."""

from __future__ import annotations

import importlib.util
import sys


REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "graphviz", "tqdm"]
MODEL_DEPENDENCIES = {
    "FSG": "fastsparsegams",
    "EBM-main / EBM-interact": "interpret",
    "ShadeTree GPU acceleration": "cupy",
}


def main():
    print(sys.executable)
    failed = False
    for package in REQUIRED:
        available = importlib.util.find_spec(package) is not None
        print(f"{package:24s} {'OK' if available else 'MISSING'}")
        failed |= not available
    for label, package in MODEL_DEPENDENCIES.items():
        available = importlib.util.find_spec(package) is not None
        print(f"{label:24s} {'OK' if available else 'MISSING'} ({package})")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
