#!/usr/bin/env bash
# Starts matricxon detached (Linux and macOS); see stop.sh / status.sh. On Debian/Ubuntu prefer
# scripts/matricxon.service.
set -euo pipefail

# Colored like uvicorn's "INFO:" lines (plain when not a terminal).
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else G=""; R=""; N=""; fi
info() { printf '%sINFO:%s %s\n' "$G" "$N" "$*"; }
err() { printf '%sERROR:%s %s\n' "$R" "$N" "$*"; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

HOST="${MATRICXON_HOST:-0.0.0.0}"
PORT="${MATRICXON_PORT:-8420}"
PID_FILE="run/matricxon.pid"
# Holds the launched pid until the app writes PID_FILE (imports and the cache scan take a while on a
# cold start), so a second start.sh waits for that server instead of launching another.
STARTING_FILE="run/matricxon.starting"
LOG_FILE="logs/matricxon.log"
# MATRICXON_START_TIMEOUT (env, else .env): the app indexes saved chats until 5 s before it, then
# answers and continues in the background; this script waits it plus a grace for the rest of startup.
START_TIMEOUT="${MATRICXON_START_TIMEOUT:-$(grep -E '^MATRICXON_START_TIMEOUT=' .env 2>/dev/null | tail -1 | cut -d= -f2 || true)}"
START_TIMEOUT="${START_TIMEOUT:-90}"
START_WAIT=$(( ${START_TIMEOUT%.*} + 15 ))

mkdir -p run logs

# Rotate on start: keep MATRICXON_LOG_KEEP old logs once the file passes MATRICXON_LOG_MAX_MB.
rotate_log() {
    local max=$(( ${MATRICXON_LOG_MAX_MB:-50} * 1024 * 1024 )) keep="${MATRICXON_LOG_KEEP:-5}" i
    [[ -f "$LOG_FILE" && "$(stat -c %s "$LOG_FILE" 2>/dev/null || stat -f %z "$LOG_FILE")" -gt $max ]] || return 0
    for ((i = keep - 1; i >= 1; i--)); do
        [[ -f "$LOG_FILE.$i" ]] && mv -f "$LOG_FILE.$i" "$LOG_FILE.$((i + 1))"
    done
    mv -f "$LOG_FILE" "$LOG_FILE.1"
}

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    info "matricxon is already running (pid $(cat "$PID_FILE"))"
    exit 1
fi

if [[ -f "$STARTING_FILE" ]] && kill -0 "$(cat "$STARTING_FILE")" 2>/dev/null; then
    SERVER_PID="$(cat "$STARTING_FILE")"
    info "matricxon is still starting (pid $SERVER_PID), waiting for it"
else
    rm -f "$PID_FILE" "$STARTING_FILE"
    rotate_log

    if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
        err "port $PORT is already in use - is another matricxon running (e.g. pAIring's external/matricxon)?"
        info "Stop that one first, or start this one on another port: MATRICXON_PORT=8421 scripts/start.sh"
        exit 1
    fi

    if command -v setsid >/dev/null 2>&1; then
        LAUNCHER=(setsid)
    else
        LAUNCHER=(nohup)
    fi

    MATRICXON_PID_FILE="$PROJECT_DIR/$PID_FILE" "${LAUNCHER[@]}" .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" >> "$LOG_FILE" 2>&1 < /dev/null &
    SERVER_PID=$!
    disown
    echo "$SERVER_PID" > "$STARTING_FILE"
fi

for ((i = 0; i < START_WAIT * 4; i++)); do
    [[ -f "$PID_FILE" ]] && break
    # Died during startup (bad config, port taken, import error...): stop waiting.
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 0.25
done

if [[ ! -f "$PID_FILE" ]]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        # Kept STARTING_FILE: a retry waits for this server instead of starting a second one.
        err "matricxon did not finish starting within ${START_WAIT}s (pid $SERVER_PID still running) - check $LOG_FILE"
    else
        rm -f "$STARTING_FILE"
        err "matricxon exited during startup - last lines of $LOG_FILE:"
        tail -n 5 "$LOG_FILE"
    fi
    exit 1
fi
rm -f "$STARTING_FILE"

info "matricxon started (pid $(cat "$PID_FILE")) on $HOST:$PORT, logging to $LOG_FILE"
