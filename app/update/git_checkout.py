"""The git commands of a self-update (self_update.py), run in this install's folder. With a source
named by the admin (`Source`), it fetches that repository through its own `update-source` remote;
`origin` and the tracked branch are never changed, and the token reaches git only through its
environment."""

import base64
import os
import re
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_PERCENT = re.compile(
    r"(Receiving objects|Resolving deltas|Counting objects|Compressing objects):\s+(\d+)%"
)


REMOTE = "update-source"


class GitError(RuntimeError):
    """A git command failed; the message is its last output line, or what to check for a refused
    private repository."""

    _ACCESS = (
        "could not read Username",
        "Authentication failed",
        "Repository not found",
        "Permission denied",
    )

    def __init__(self, message: str) -> None:
        if any(text in message for text in self._ACCESS):
            message = (
                "The repository refused access. Check the repository and the access token (it needs"
                " read access), or for SSH this machine's key."
            )
        elif "couldn't find remote ref" in message:
            message = "That branch or tag is not in the repository."
        super().__init__(message)


@dataclass(frozen=True)
class Source:
    """A repository to update from instead of the checkout's upstream."""

    url: str
    branch: str
    token: str | None = None

    @property
    def ref(self) -> str:
        return f"refs/remotes/{REMOTE}/{self.branch}"

    def env(self) -> dict[str, str]:
        """The token as an HTTP header for this URL only (in the environment: ps shows argv)."""
        if not self.token:
            return {}
        user = "oauth2" if "gitlab" in (urlparse(self.url).hostname or "") else "x-access-token"
        basic = base64.b64encode(f"{user}:{self.token}".encode()).decode()
        return {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"http.{self.url}.extraHeader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
        }


class GitCheckout:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._source: Source | None = None

    def use(self, source: Source | None) -> None:
        """Fetch from `source` from now on; None: the checkout's own upstream."""
        self._source = source

    @property
    def _upstream(self) -> str:
        return self._source.ref if self._source else "@{u}"

    def is_checkout(self) -> bool:
        return (self.folder / ".git").exists()

    def head(self) -> str:
        return self._run("rev-parse", "--short", "HEAD")

    def fetch(self, on_percent: Callable[[str, int], None]) -> None:
        """`git fetch` of the upstream; `on_percent(activity, percent)` as git reports it."""
        args = ["fetch", "--progress"]
        if self._source:
            try:
                self._run("remote", "set-url", REMOTE, self._source.url)
            except GitError:
                self._run("remote", "add", "--no-fetch", REMOTE, self._source.url)
            args += [REMOTE, f"+{self._source.branch}:{self._source.ref}"]
        process = subprocess.Popen(
            ["git", *args],
            cwd=self.folder,
            env=self._env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        tail = ""
        for line in self._lines(process.stderr):
            tail = line or tail
            if match := _PERCENT.search(line):
                on_percent(match[1].lower(), int(match[2]))
        if process.wait() != 0:
            raise GitError(f"git fetch failed: {tail}")

    def incoming(self) -> list[str]:
        """Files the fetched upstream changes, compared with this checkout."""
        out = self._run(
            "diff", "--name-only", f"HEAD...{self._upstream}"
        )  # upstream's own changes only
        return out.splitlines() if out else []

    def fast_forward(self) -> None:
        self._run("merge", "--ff-only", self._upstream)

    def _run(self, *args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=self.folder, capture_output=True, text=True, env=self._env()
        )
        if done.returncode != 0:
            last = (done.stderr or done.stdout).strip().splitlines()
            raise GitError(f"git {args[0]} failed: {last[-1] if last else done.returncode}")
        return done.stdout.strip()

    def _env(self) -> dict[str, str]:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"}
        return {**env, **self._source.env()} if self._source else env

    @staticmethod
    def _lines(stream) -> Iterator[str]:
        """git rewrites its progress line with \\r, so split on both."""
        buffer = ""
        while chunk := stream.read(256):
            buffer += chunk
            *lines, buffer = re.split(r"[\r\n]", buffer)
            yield from lines
        if buffer:
            yield buffer
