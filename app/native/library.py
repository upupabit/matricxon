"""Builds and loads matricxon's native kernel library (`app/native/src/*.c`), compiled on first use
with the system C compiler: no CMake, no extra pip dependency. `-march=native` targets the CPU the
process runs on, which is why it is built locally, not shipped prebuilt.

Rebuilt whenever a source file is newer than the library. The build writes a temp file and renames
it into place, so two matricxon instances starting together (pAIring can run several) never load a
half-written library.
"""

import ctypes
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.native.compiler_flags import BuildVariant, CompilerFlags

logger = logging.getLogger(__name__)

_NATIVE_DIR = Path(__file__).resolve().parent


class NativeBuildError(RuntimeError):
    pass


class NativeKernelLibrary:
    LIBRARY_NAME = "libmatricxon_kernels.so"

    def __init__(
        self,
        source_dir: Path = _NATIVE_DIR / "src",
        build_dir: Path = _NATIVE_DIR / "build",
        compiler: str | None = None,
        flags: CompilerFlags | None = None,
        wide_simd: bool = False,
    ) -> None:
        self._source_dir = source_dir
        self._build_dir = build_dir
        self._wide_simd = wide_simd
        self._flags = flags or CompilerFlags(wide_simd=wide_simd)
        self._compiler = compiler or os.environ.get("CC", self._flags.default_compiler())

    @property
    def library_path(self) -> Path:
        """A separate file for the wide-SIMD build (Settings.wide_simd), so switching rebuilds."""
        name = (
            self.LIBRARY_NAME.replace(".so", "_wide.so") if self._wide_simd else self.LIBRARY_NAME
        )
        return self._build_dir / name

    def _sources(self) -> list[Path]:
        return sorted(self._source_dir.glob("*.c"))

    def _is_stale(self) -> bool:
        if not self.library_path.exists():
            return True
        built_at = self.library_path.stat().st_mtime
        # The flags too: a flag change (compiler_flags.py) needs a rebuild like a source change.
        flags = Path(__file__).with_name("compiler_flags.py")
        inputs = [*self._sources(), *self._source_dir.glob("*.h"), flags]
        return any(path.stat().st_mtime > built_at for path in inputs)

    def _compile(self, variant: BuildVariant, output: str) -> subprocess.CompletedProcess:
        command = [
            self._compiler,
            *variant.flags,
            "-o",
            output,
            *map(str, self._sources()),
            *variant.link_flags,
            "-lm",
        ]
        return subprocess.run(command, capture_output=True, text=True)

    def build(self) -> Path:
        """Compiles every `src/*.c` into one shared library, trying each of CompilerFlags.variants
        in turn (OpenMP first, then single-threaded; app/native/compiler_flags.py). Raises
        NativeBuildError (with every variant's compiler output) if no compiler is installed or every
        variant fails."""
        if shutil.which(self._compiler) is None:
            raise NativeBuildError(f"C compiler {self._compiler!r} not found on PATH")
        self._build_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self._build_dir, suffix=".so.tmp")
        os.close(fd)
        failures = []
        for variant in self._flags.variants():
            result = self._compile(variant, tmp_name)
            if result.returncode == 0:
                os.chmod(tmp_name, 0o755)
                os.replace(tmp_name, self.library_path)
                logger.info("built native kernels (%s): %s", variant.name, self.library_path)
                return self.library_path
            logger.info("native kernel build variant %r failed, trying the next one", variant.name)
            failures.append(f"[{variant.name}]\n{result.stderr}")
        Path(tmp_name).unlink(missing_ok=True)
        raise NativeBuildError("native kernel build failed:\n" + "\n".join(failures))

    def load(self) -> ctypes.CDLL:
        """Builds first if needed, then loads. ctypes releases the GIL for the whole native call, so
        a running kernel never blocks the asyncio event loop's thread."""
        if self._is_stale():
            self.build()
        # Idle OpenMP threads sleep instead of busy-spinning between calls. libgomp reads this once
        # when first loaded (here; PyTorch ships its own copy, already running and unaffected).
        # Incident: spinning native threads competing with PyTorch's on 4 CPUs likely
        # pushed the dev machine into a thermal shutdown during a benchmark.
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        return ctypes.CDLL(str(self.library_path))
