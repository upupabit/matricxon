"""CPU counts for the thread default. Logical CPUs win on the dev machine (2 cores, 4 threads,
2026-10-19, with the kernels' chunked scheduling): on a ~2,100-token chat a decode step took 1.05 s
with 4 threads vs 1.19 s with 2, and prompt passes were faster too. Physical cores are kept for
a later per-machine choice."""

import os
from pathlib import Path

_TOPOLOGY = Path("/sys/devices/system/cpu")


class CpuCores:
    @staticmethod
    def logical() -> int:
        """CPUs this process may run on."""
        try:
            return len(os.sched_getaffinity(0)) or 1
        except AttributeError:  # not Linux
            return os.cpu_count() or 1

    @classmethod
    def physical(cls) -> int:
        """Physical cores among this process's CPUs (Linux sysfs topology); elsewhere, the logical
        count."""
        try:
            cpus = os.sched_getaffinity(0)
        except AttributeError:
            return cls.logical()
        cores = set()
        for cpu in cpus:
            topology = _TOPOLOGY / f"cpu{cpu}" / "topology"
            try:
                package = (topology / "physical_package_id").read_text().strip()
                core = (topology / "core_id").read_text().strip()
            except OSError:
                return cls.logical()
            cores.add((package, core))
        return len(cores) or cls.logical()

    @classmethod
    def threads(cls, configured: int | None) -> int:
        """The thread count to use: the configured one, else one per logical CPU."""
        return configured or cls.logical()
