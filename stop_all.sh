#!/usr/bin/env bash
# Stop services started by start_all.sh.
#   ./stop_all.sh              stop all: game server, OpenJev service, LLM container
#   ./stop_all.sh game jev     stop only the named ones (game | jev | llm)
#   ./stop_all.sh --dry-run    show what would be stopped
set -uo pipefail

DRY=0
TARGETS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    game|jev|llm) TARGETS+=("$a") ;;
    *) echo "unknown argument: $a (use game | jev | llm | --dry-run)" >&2; exit 1 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(game jev llm)

stop_port() {  # name port
  local name=$1 port=$2 pids
  pids=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  if [ -z "$pids" ]; then echo "$name (port $port): not running"; return; fi
  echo "$name (port $port): stopping pid $pids"
  [ $DRY -eq 1 ] && return
  kill $pids
  for _ in $(seq 1 20); do
    ss -ltnH "sport = :$port" | grep -q . || { echo "$name: stopped"; return; }
    sleep 0.5
  done
  echo "$name: still running after 10s, sending SIGKILL"; kill -9 $pids 2>/dev/null
}

for t in "${TARGETS[@]}"; do
  case "$t" in
    game) stop_port "game server" 38320 ;;
    jev)  stop_port "OpenJev service" 18310 ;;  # frees ~19 GB on the RTX 5090
    llm)
      if docker ps --format '{{.Names}}' | grep -qx gomoku-llm; then
        echo "LLM container gomoku-llm: stopping"
        if [ $DRY -eq 0 ]; then docker stop gomoku-llm >/dev/null && echo "LLM: stopped"; fi
      else
        echo "LLM container gomoku-llm: not running"
      fi ;;
  esac
done
