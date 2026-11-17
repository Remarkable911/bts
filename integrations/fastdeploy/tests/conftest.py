import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "integrations/fastdeploy/src"))
sys.path.insert(0, os.environ.get("BTS_SOURCE_ROOT", str(ROOT)))
# Keep the submission root available for the shared guard without overriding a
# wheel-target BTS_SOURCE_ROOT used by adapter isolation tests.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from eval.ThreeWay.pytest_gpu_guard import (  # noqa: E402
    configure,
    prepare_environment,
    unconfigure,
)


prepare_environment()


def pytest_configure(config):
    configure(config)


def pytest_unconfigure(config):
    unconfigure(config)
