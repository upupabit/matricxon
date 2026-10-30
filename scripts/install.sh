#!/usr/bin/env bash
# Installs matricxon on Linux in this checkout, without root: Python environment, dependencies, .env and the native
# kernels. For a server that runs Matricxon on its own, e.g. a cluster node. With pAIring, install it from pAIring's
# UI instead (Settings > Inference engines), which puts it in pAIring's own folder. macOS: scripts/install_mac.sh.
#
# Usage: scripts/install.sh
#
# On a server it suggests /opt/matricxon and the user pAIring runs as, "pairing", and asks before going on otherwise
# (with no terminal it goes on).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

step() { printf '\n==> %s\n' "$1"; }
info() { printf '    %s\n' "$1"; }
warn() { printf '  ! %s\n' "$1"; }
fail() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

# Asks before going on as things are; no terminal: goes on. "No" stops here, nothing changed.
ask_continue() {
    if [[ ! -t 0 ]]; then
        info "No terminal to ask: continuing."
        return 0
    fi
    read -r -p "$1 [y/N] " answer
    case "$answer" in
        [Yy]*) ;;
        *) echo "Stopped, nothing was changed. Run the commands above, then this installer again."; exit 0 ;;
    esac
}

[[ "$(uname -s)" == "Linux" ]] || fail "this script is for Linux. On macOS, use scripts/install_mac.sh."

me="$(id -un)"
if [[ "$PROJECT_DIR" != /opt/* ]]; then
    step "Checking where matricxon is installed"
    warn "matricxon is in $PROJECT_DIR. On a server it belongs in /opt/matricxon, outside anyone's home folder."
    info "To install it there instead (an admin with sudo runs the first line):"
    info "  sudo mkdir -p /opt/matricxon && sudo chown $me: /opt/matricxon"
    info "  git clone https://github.com/upupabit/matricxon.git /opt/matricxon   (or copy this folder there)"
    info "  cd /opt/matricxon && scripts/install.sh"
    ask_continue "Install here in $PROJECT_DIR anyway?"
fi

if [[ "$me" != "pairing" ]]; then
    step "Checking the user matricxon will run as"
    warn "On a server, matricxon is best run as the user pAIring runs as, 'pairing' (no sudo), not as $me."
    info "On a cluster, every node needs that same user with the same user ID, so each can write the others' files"
    info "on the shared storage. Find the ID on the pAIring server with: id -u pairing"
    if id pairing >/dev/null 2>&1; then
        info "To switch:  sudo chown -R pairing: $PROJECT_DIR && sudo -u pairing $PROJECT_DIR/scripts/install.sh"
    else
        info "To create it and install as it (an admin with sudo runs these):"
        info "  sudo useradd -m -u <that ID> -s /bin/bash pairing"
        info "  sudo chown -R pairing: $PROJECT_DIR"
        info "  sudo -u pairing $PROJECT_DIR/scripts/install.sh"
    fi
    ask_continue "Continue as $me anyway?"
fi

step "Checking the C compiler (for the fast native kernels)"
if command -v gcc >/dev/null 2>&1; then
    echo "ok: $(gcc --version | head -1)"
else
    warn "gcc not found: matricxon will use its much slower Numba kernels. Install it with: sudo apt install build-essential"
fi

step "Finding Python 3.11 or newer"
PYTHON=""
for candidate in "${PYTHON_BIN:-}" python3.13 python3.12 python3.11 python3; do
    [[ -z "$candidate" ]] && continue
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
[[ -n "$PYTHON" ]] || fail "no Python 3.11+ found. Install one with: sudo apt install python3 python3-venv"
echo "ok: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

step "Creating the Python environment (.venv)"
if [[ -x .venv/bin/python ]]; then
    echo "ok: .venv already exists - reusing it"
else
    "$PYTHON" -m venv .venv || fail "could not create .venv. On Debian/Ubuntu: sudo apt install python3-venv"
fi

step "Installing dependencies (PyTorch is a large download - this can take several minutes)"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

step "Writing .env"
if [[ -f .env ]]; then
    echo "ok: .env already exists - left unchanged"
else
    cat > .env <<'EOF'
# matricxon settings - see app/config.py for every option. Real environment variables win over this file.
MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE=true
MATRICXON_GEMV_BACKEND=native
MATRICXON_MEMORY_SAFETY_MARGIN=1.2
MATRICXON_LOG_LEVEL=0
# Where models are stored (default: ./data/models inside this folder; on a cluster, the shared storage):
# MATRICXON_MODELS_DIR=/mnt/shared/models
# CPU threads (default: every core) - fewer runs cooler:
# MATRICXON_TORCH_THREADS=4
EOF
    echo "ok: created .env (native kernels on, memory safety margin 1.2, log level 0)"
fi

step "Building the native C kernels"
PYTHONPATH="$PROJECT_DIR" .venv/bin/python - <<'EOF'
from app.native.gemm import NativeGemm
from app.native.library import NativeBuildError, NativeKernelLibrary

try:
    library = NativeKernelLibrary().load()
except NativeBuildError as exc:
    print(f"The native kernels did not build - matricxon will use its Numba kernels instead.\n{exc}")
else:
    info = NativeGemm(library, 1).build_info
    threading = "multi-threaded" if "openmp=yes" in info else "single-threaded"
    print(f"ok: native kernels built ({threading}; {info})")
EOF

step "Done"
cat <<EOF
Start matricxon:   scripts/start.sh      (then: scripts/status.sh, scripts/stop.sh)
Log file:          logs/matricxon.log
Port:              8420

To start it with the server after every reboot: edit scripts/matricxon.service (paths, User=$me),
then: sudo cp scripts/matricxon.service /etc/systemd/system/ && sudo systemctl enable --now matricxon

To use it from pAIring on another machine: Settings > Inference engines > Matricxon, Remote mode,
and add http://<this-server>:8420
EOF
