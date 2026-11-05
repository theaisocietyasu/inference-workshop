#!/usr/bin/env bash
# One command for the workshop: installs llama.cpp if needed, starts llama-server
# with its web UI, runs the inference experiments, then keeps the server up.
#
# Environment (all optional; setup.sh lists more):
#   PORT           server port (default: first free port from 8080)
#   HOST           bind address (default: 127.0.0.1)
#   PARALLEL       request slots the server batches together (default: 4)
#   CTX            total context length shared by the slots (default: 8192)
#   SKIP_METRICS   1 starts the server without the experiments
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLAMA_HOME="${LLAMA_HOME:-$REPO_DIR/.llama}"
HOST="${HOST:-127.0.0.1}"
PARALLEL="${PARALLEL:-4}"
CTX="${CTX:-8192}"

command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }

"$REPO_DIR/setup.sh"
# shellcheck disable=SC1091
source "$LLAMA_HOME/env"

if [ -z "${PORT:-}" ]; then
  PORT="$(python3 - <<'EOF'
import socket
for port in range(8080, 8200):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
EOF
)"
fi

ngl=0
[ "$BACKEND" != cpu ] && ngl=999
logfile="$LLAMA_HOME/server-$PORT.log"

echo "[run] starting llama-server ($BACKEND) on $HOST:$PORT, log: $logfile"
"$LLAMA_BIN/llama-server" -m "$MODEL" --host "$HOST" --port "$PORT" \
  -np "$PARALLEL" -c "$CTX" -ngl "$ngl" --metrics >"$logfile" 2>&1 &
server_pid=$!
trap 'kill $server_pid 2>/dev/null || true' EXIT INT TERM

url="http://127.0.0.1:$PORT"
for _ in $(seq 1 180); do
  curl -sf "$url/health" >/dev/null 2>&1 && break
  kill -0 "$server_pid" 2>/dev/null || { tail -20 "$logfile"; echo "[run] server exited" >&2; exit 1; }
  sleep 1
done
curl -sf "$url/health" >/dev/null || { tail -20 "$logfile"; echo "[run] server did not become ready" >&2; exit 1; }

if [ "${SKIP_METRICS:-0}" != 1 ]; then
  python3 "$REPO_DIR/metrics.py" --url "$url" --pid "$server_pid" --backend "$BACKEND" --parallel "$PARALLEL"
fi

cat <<EOF

Web UI and playground:  $url
  On a remote pod, forward the port first:
    VS Code Remote-SSH: Ports panel, forward $PORT
    plain ssh:          ssh -L $PORT:127.0.0.1:$PORT <your usual ssh target>
  Rerun the experiments:  python3 metrics.py --url $url
  Prometheus counters:    $url/metrics
Press Ctrl-C to stop the server.
EOF
wait "$server_pid"
