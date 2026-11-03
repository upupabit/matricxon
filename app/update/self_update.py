"""Updates this node from its own GitHub source when an admin asks (POST /api/update): git fetch,
fast-forward, pip install when requirements.txt changed, then a graceful restart. The source is the
checkout's own upstream, or a repository the admin names (e.g. a partner's private one), which
needs the admin secret: without one, anyone reaching the port could make the node run any code.
Runs in a thread; `status()` reports the step and percent, so pAIring can update a cluster one node
at a time. Only for a git checkout, and not in a container (update its image instead).
"""

import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from app.update.git_checkout import GitCheckout, GitError, Source

PROJECT_DIR = Path(__file__).resolve().parents[2]


class UpdateRefusedError(RuntimeError):
    """Shown to the caller; nothing was changed."""


class PipError(RuntimeError):
    """pip failed; the message is its last output line."""


@dataclass
class UpdateStatus:
    state: str = "idle"  # idle, updating, up_to_date, restarting, error
    step: int = 0
    total_steps: int = 3
    label: str = ""
    percent: int | None = None
    error: str | None = None
    commit_before: str | None = None
    commit_after: str | None = None


class SelfUpdater:
    RESTART_DELAY_SECONDS = 1.0

    def __init__(
        self,
        restart: Callable[[], None],
        git: GitCheckout | None = None,
        pip: Callable[[Path], None] | None = None,
        in_container: Callable[[], bool] = lambda: Path("/.dockerenv").exists(),
        has_admin_secret: Callable[[], bool] = lambda: False,
    ) -> None:
        self._has_admin_secret = has_admin_secret
        self._restart = restart
        self._git = git or GitCheckout(PROJECT_DIR)
        self._pip = pip or self._pip_install
        self._in_container = in_container
        self._status = UpdateStatus()
        self._lock = threading.Lock()

    def status(self) -> dict:
        return asdict(self._status)

    def start(self, source: Source | None = None) -> dict:
        """`source`: a repository to update from instead of this checkout's upstream."""
        with self._lock:
            if self._status.state in ("updating", "restarting"):
                raise UpdateRefusedError("An update of this node is already running.")
            if self._in_container():
                raise UpdateRefusedError("This node runs in a container: update its image instead.")
            if not self._git.is_checkout():
                raise UpdateRefusedError(
                    "This install is not a git checkout: update it on its machine."
                )
            if source and not self._has_admin_secret():
                raise UpdateRefusedError(
                    "Updating from a named repository needs MATRICXON_ADMIN_SECRET on this node."
                )
            self._git.use(source)
            self._status = UpdateStatus(state="updating", step=1, label="Downloading changes")
        threading.Thread(target=self._run, name="self-update", daemon=True).start()
        return self.status()

    def run(self) -> None:
        """The whole update, in the caller's thread (tests)."""
        self._run()

    def _run(self) -> None:
        status = self._status
        try:
            status.commit_before = self._git.head()
            self._git.fetch(self._on_percent)
            changed = self._git.incoming()
            if not changed:
                status.state, status.percent = "up_to_date", None
                return
            self._step(2, "Applying changes")
            self._git.fast_forward()
            if "requirements.txt" in changed:
                self._step(3, "Installing dependencies")
                self._pip(PROJECT_DIR / "requirements.txt")
            status.commit_after = self._git.head()
            status.state = "restarting"
            threading.Timer(self.RESTART_DELAY_SECONDS, self._restart).start()
        except (GitError, PipError, OSError) as exc:
            status.state, status.error = "error", str(exc)

    def _on_percent(self, activity: str, percent: int) -> None:
        self._status.label = f"Downloading changes — {activity}"
        self._status.percent = percent

    def _step(self, step: int, label: str) -> None:
        self._status.step, self._status.label, self._status.percent = step, label, None

    @staticmethod
    def _pip_install(requirements: Path) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            last = (done.stderr or done.stdout).strip().splitlines()
            raise PipError(f"pip install failed: {last[-1] if last else done.returncode}")
