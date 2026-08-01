"""Runtime dependency checks for optional framework integrations.

Generated with AI assistance and reviewed/adapted for thesis reproducibility.
"""

from __future__ import annotations

from importlib import import_module


FRAMEWORK_IMPORTS: dict[str, tuple[str, ...]] = {
    "langgraph": ("langgraph", "langchain_openai"),
    "autogen": ("autogen_agentchat", "autogen_ext.models.openai"),
    "crewai": ("crewai",),
    "metagpt": ("metagpt",),
}


def detect_missing_framework_imports(frameworks: list[str]) -> dict[str, list[str]]:
    """Return missing import paths per requested framework."""

    missing: dict[str, list[str]] = {}
    for framework in frameworks:
        required_modules = FRAMEWORK_IMPORTS.get(framework, ())
        missing_modules: list[str] = []
        for module_name in required_modules:
            try:
                import_module(module_name)
            except Exception:
                missing_modules.append(module_name)
        if missing_modules:
            missing[framework] = missing_modules
    return missing


def assert_framework_runtime_ready(frameworks: list[str]) -> None:
    """Raise a clear error when a requested framework lacks runtime deps."""

    missing = detect_missing_framework_imports(frameworks)
    if not missing:
        return
    details = "; ".join(f"{framework}: {', '.join(modules)}" for framework, modules in sorted(missing.items()))
    raise RuntimeError(
        "Missing optional framework dependencies for native execution. "
        f"Missing modules -> {details}. "
        "Install optional dependencies with: `uv sync --extra dev --extra frameworks` "
        "or `pip install -e \".[dev,frameworks]\"`. "
        "For MetaGPT in particular, use an isolated environment because its dependency "
        "constraints can conflict with this project's numpy/pandas stack. "
        "On Apple Silicon/macOS, prefer the provided Linux/x86 Docker runtime."
    )
