"""Runtime readiness checks for optional framework dependencies.

This script was generated with AI assistance and reviewed/adapted for
transparent thesis reproducibility checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.runtime_checks import FRAMEWORK_IMPORTS, detect_missing_framework_imports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check optional framework runtime dependencies.")
    parser.add_argument(
        "--frameworks",
        nargs="*",
        default=list(FRAMEWORK_IMPORTS.keys()),
        choices=sorted(FRAMEWORK_IMPORTS.keys()),
        help="Framework subset to check. Default: all.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any requested framework is missing imports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frameworks = list(dict.fromkeys(args.frameworks))
    missing = detect_missing_framework_imports(frameworks)

    print("=== Runtime Dependency Check ===")
    for framework in frameworks:
        required_modules = FRAMEWORK_IMPORTS.get(framework, ())
        missing_modules = set(missing.get(framework, ()))
        state = "OK" if not missing_modules else "MISSING"
        modules_display = ", ".join(required_modules) if required_modules else "-"
        print(f"- {framework:<10} {state:<8} (imports: {modules_display})")
        if missing_modules:
            print(f"  missing: {', '.join(sorted(missing_modules))}")

    print("")
    if not missing:
        print("All optional framework dependencies are available.")
    else:
        print("Some optional dependencies are missing.")
        print("LangGraph-only validation can still run; full 4-framework baseline will use fallbacks where missing.")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
