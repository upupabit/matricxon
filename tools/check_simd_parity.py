"""Checks that every SIMD build of the native kernels gives bit-identical results to plain C,
including CPUs this machine lacks, under QEMU user-mode emulation. No root needed: unpack
`qemu-user` (from `apt-get download`) and, for ARM, `gcc-14-aarch64-linux-gnu` with its
`*-arm64-cross` libraries into one folder and pass it as --root.

    .venv/bin/python tools/check_simd_parity.py --root /path/to/unpacked [--extra-flag ...]

Each variant builds tools/simd_parity.c with all kernel sources, runs it (natively, or under qemu
with a CPU model that has the instructions) and compares its output with the plain-C x86 build.
AVX-VNNI is build-checked only: QEMU's TCG doesn't emulate it.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCES = sorted(str(p) for p in (REPO / "app/native/src").glob("*.c"))
HARNESS = str(REPO / "tools/simd_parity.c")


@dataclass
class Variant:
    name: str
    arm: bool
    flags: tuple[str, ...]
    qemu_cpu: str | None = None  # None: run natively; "-": build only


VARIANTS = (
    Variant("plain C (x86)", False, ("-march=x86-64",)),
    Variant("SSSE3 (this CPU)", False, ("-march=native",)),
    Variant("default build on Haswell", False, ("-march=haswell",), "Haswell"),
    Variant("wide AVX2", False, ("-march=haswell", "-DMX_WIDE_SIMD"), "Haswell"),
    Variant("wide AVX-VNNI", False, ("-march=alderlake", "-DMX_WIDE_SIMD"), "-"),
    Variant("wide AVX-512 VNNI", False, ("-march=sapphirerapids", "-DMX_WIDE_SIMD"), "-"),
    Variant("plain C (ARM64)", True, ("-march=armv8-a",), "max"),
    Variant("wide NEON dotprod", True, ("-march=armv8.2-a+dotprod", "-DMX_WIDE_SIMD"), "max"),
)


class ParityCheck:
    def __init__(self, root: Path, out_dir: Path) -> None:
        self._root = root
        self._out = out_dir
        self.extra: list[str] = []
        self._env = {**os.environ, "LD_LIBRARY_PATH": str(root / "usr/lib/x86_64-linux-gnu")}
        self._env["PATH"] = f"{root / 'usr/bin'}:{os.environ['PATH']}"

    def _compiler(self, arm: bool) -> list[str]:
        if not arm:
            return ["gcc"]
        r = self._root
        return [
            str(r / "usr/bin/aarch64-linux-gnu-gcc-14"),
            f"--sysroot={r}",
            f"-B{r}/usr/lib/gcc-cross/aarch64-linux-gnu/14",
            f"-B{r}/usr/aarch64-linux-gnu/lib",
        ]

    def run(self, variant: Variant) -> str | None:
        binary = self._out / variant.name.replace(" ", "_").replace("(", "").replace(")", "")
        # -ffp-contract=off as app/native/compiler_flags.py builds it.
        base = ["-O3", "-fopenmp", "-ffp-contract=off"]
        cmd = [*self._compiler(variant.arm), *base, *variant.flags, *self.extra]
        cmd += ["-o", str(binary), HARNESS, *SOURCES]
        subprocess.run([*cmd, "-lm"], check=True, env=self._env, capture_output=True, text=True)
        if variant.qemu_cpu == "-":
            return None
        if variant.qemu_cpu is None:
            run = [str(binary)]
        else:
            qemu = "qemu-aarch64" if variant.arm else "qemu-x86_64"
            run = [str(self._root / "usr/bin" / qemu), "-cpu", variant.qemu_cpu]
            if variant.arm:
                run += ["-L", str(self._root / "usr/aarch64-linux-gnu")]
            run.append(str(binary))
        result = subprocess.run(run, check=True, env=self._env, capture_output=True, text=True)
        return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="unpacked qemu-user + arm64 gcc")
    parser.add_argument("--out", type=Path, default=Path("/tmp/simd_parity"))
    parser.add_argument("--extra-flag", action="append", default=[], help="added to every build")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    check = ParityCheck(args.root, args.out)
    check.extra = args.extra_flag
    reference = check.run(VARIANTS[0])
    lines = reference.count("\n")
    values = reference.count(" ") - 4 * lines
    print(f"reference ({VARIANTS[0].name}): {lines} result lines, {values} values")
    failed = 0
    for variant in VARIANTS[1:]:
        output = check.run(variant)
        if output is None:
            print(f"  {variant.name}: builds (not run: no emulator for it)")
        elif output == reference:
            print(f"  {variant.name}: identical")
        else:
            failed += 1
            pairs = zip(reference.splitlines(), output.splitlines(), strict=False)
            diff = [a for a, b in pairs if a != b]
            first = diff[0][:60] if diff else "(length)"
            print(f"  {variant.name}: DIFFERS in {len(diff)} line(s), first: {first}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
