"""The default thread count is one per logical CPU; physical cores are counted too."""

from app.runtime.cpu_cores import CpuCores


def test_physical_cores_are_at_most_the_logical_cpus() -> None:
    assert 1 <= CpuCores.physical() <= CpuCores.logical()


def test_a_configured_count_wins() -> None:
    assert CpuCores.threads(3) == 3
    assert CpuCores.threads(None) == CpuCores.logical()
