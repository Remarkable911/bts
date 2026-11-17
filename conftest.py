"""Repository-wide pytest guard for formal single-GPU benchmarks."""

from eval.ThreeWay.pytest_gpu_guard import configure, prepare_environment, unconfigure


prepare_environment()


def pytest_configure(config):
    configure(config)


def pytest_unconfigure(config):
    unconfigure(config)
