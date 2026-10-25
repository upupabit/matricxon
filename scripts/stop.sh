#!/usr/bin/env bash
# Stops a matricxon instance started via scripts/start.sh.
set -euo pipefail

# Colored like uvicorn's "INFO:" lines (plain when not a terminal).
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else G=""; R=""; N=""; fi
info() { printf '%sINFO:%s %s\n' "$G" "$N" "$*"; }
err() { printf '%sERROR:%s %s\n' "$R" "$N" "$*"; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PID_FILE="run/matricxon.pid"
# Written by start.sh until the app writes PID_FILE: a server still starting is stopped too.
STARTING_FILE="run/matricxon.starting"

if [[ ! -f "$PID_FILE" && -f "$STARTING_FILE" ]] && kill -0 "$(cat "$STARTING_FILE")" 2>/dev/null; then
    PID_FILE="$STARTING_FILE"
fi

if [[ ! -f "$PID_FILE" ]]; then
    rm -f "$STARTING_FILE"
    info "matricxon is not running (no pid file)"
    exit 0
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
    info "matricxon is not running (stale pid file, removing it)"
    rm -f "$PID_FILE"
    exit 0
fi

kill -TERM "$PID" 2>/dev/null || true

# Up to 60 s: shutdown saves cached chats to disk first, and a big one takes a while.
for _ in $(seq 1 120); do
    if ! kill -0 "$PID" 2>/dev/null; then
        rm -f "$PID_FILE" "$STARTING_FILE"
        info "matricxon stopped (pid $PID)"
        exit 0
    fi
    sleep 0.5
done

info "matricxon did not stop gracefully, sending SIGKILL"
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE" "$STARTING_FILE"
