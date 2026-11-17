"""Validate package provenance and prepare FastDeploy child processes."""

from importlib.metadata import entry_points
import os
from pathlib import Path
import sys


_FASTDEPLOY_PLUGIN_GROUP = "fastdeploy.model_register_plugins"
_FASTDEPLOY_PLUGIN_NAME = "bts"
_FASTDEPLOY_PLUGIN_VALUE = "bts_fastdeploy.plugin:register"


def _required_source_root():
    value = os.environ.get("BTS_SOURCE_ROOT")
    return Path(value).expanduser().resolve() if value else None


def validate_package_provenance(provenance, source_root=None):
    """Normalize package paths and enforce an optional submission root."""
    if not isinstance(provenance, dict):
        raise RuntimeError("BTS package provenance must be an object")
    required = ("bts_file", "bts_fastdeploy_file")
    missing = [name for name in required if not provenance.get(name)]
    if missing:
        raise RuntimeError(
            "BTS package provenance is missing: " + ", ".join(missing)
        )

    normalized = {
        name: str(Path(provenance[name]).expanduser().resolve()) for name in required
    }
    missing_files = [name for name, value in normalized.items() if not Path(value).is_file()]
    if missing_files:
        raise RuntimeError(
            "BTS package provenance files do not exist: "
            + ", ".join(missing_files)
        )
    root = _required_source_root() if source_root is None else source_root
    if root is not None:
        root = Path(root).expanduser().resolve()
        expected = {
            "bts_file": (root / "bts",),
            # A wheel --target install uses the first layout; the submitted
            # source tree uses the second one. Both keep the active package
            # strictly inside BTS_SOURCE_ROOT.
            "bts_fastdeploy_file": (
                root / "bts_fastdeploy",
                root / "integrations/fastdeploy/src/bts_fastdeploy",
            ),
        }
        violations = []
        for name, directories in expected.items():
            inside = False
            for directory in directories:
                try:
                    Path(normalized[name]).relative_to(directory)
                    inside = True
                    break
                except ValueError:
                    pass
            if not inside:
                violations.append(
                    f"{name}={normalized[name]!r} is outside "
                    f"{[str(directory) for directory in directories]!r}"
                )
        if violations:
            raise RuntimeError(
                "BTS_SOURCE_ROOT package provenance check failed: "
                + "; ".join(violations)
            )
    normalized["required_source_root"] = str(root) if root is not None else None
    return normalized


def package_provenance():
    """Return the actual launcher/worker package files, failing closed if pinned."""
    import bts
    import bts_fastdeploy

    return validate_package_provenance(
        {
            "bts_file": bts.__file__,
            "bts_fastdeploy_file": bts_fastdeploy.__file__,
        }
    )


def validate_fastdeploy_plugin_entry_point():
    """Require the entry point that a freshly spawned GPU worker discovers.

    Adding the submission sources to ``PYTHONPATH`` makes the modules
    importable, but it does not expose the entry-point metadata declared in
    ``pyproject.toml``.  FastDeploy discovers plugins from installed
    distribution metadata in every GPU worker, so fail before model loading
    when the submission was not installed into this exact interpreter.
    """
    matches = [
        candidate
        for candidate in entry_points(group=_FASTDEPLOY_PLUGIN_GROUP)
        if candidate.name == _FASTDEPLOY_PLUGIN_NAME
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "FastDeploy GPU workers cannot discover the BTS plugin entry "
            f"point: expected exactly one {_FASTDEPLOY_PLUGIN_NAME!r} entry "
            f"in {_FASTDEPLOY_PLUGIN_GROUP!r}, found {len(matches)}. "
            "PYTHONPATH is not sufficient; install the submission into the "
            "environment selected by PYTHON_BIN, for example: "
            f"{sys.executable} -m pip install --no-deps -e "
            "<submission-root>"
        )
    selected = matches[0]
    if selected.value != _FASTDEPLOY_PLUGIN_VALUE:
        raise RuntimeError(
            "FastDeploy BTS plugin entry point targets the wrong callable: "
            f"expected {_FASTDEPLOY_PLUGIN_VALUE!r}, found {selected.value!r}"
        )
    distribution = getattr(selected, "dist", None)
    return {
        "group": _FASTDEPLOY_PLUGIN_GROUP,
        "name": selected.name,
        "value": selected.value,
        "distribution": getattr(distribution, "name", None),
        "distribution_version": getattr(distribution, "version", None),
    }


def prepare_workers():
    import bts
    import torch

    provenance = package_provenance()
    validate_fastdeploy_plugin_entry_point()
    # Put the installed submission package (or its editable source root) first
    # for child workers; the single wheel contains both the ``bts`` package
    # namespace and this BTS adapter.
    root = str(Path(bts.__file__).resolve().parents[1])
    paths = [root, *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))

    # A venv can inherit Torch/CUDA wheels read-only from another environment.
    # Native loaders in a fresh worker do not search Python's sys.path for .so's.
    libraries = [str(Path(torch.__file__).parent / "lib")]
    for entry in sys.path:
        libraries.extend(str(p) for p in (Path(entry) / "nvidia").glob("*/lib") if p.is_dir())
    libraries.extend(filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)))
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(libraries))
    return provenance
