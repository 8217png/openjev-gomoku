#!/usr/bin/env bash
# Start: LLM opponent (llama.cpp, GPU0 4060Ti) -> OpenJev decision service (GPU1 5090) -> game server.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
mkdir -p logs
if ! docker ps --format '{{.Names}}' | grep -q '^gomoku-llm$'; then
  docker start gomoku-llm 2>/dev/null || docker run -d --name gomoku-llm --gpus '"device=0"' -p 18300:8080 \
    -v "${LLM_GGUF_DIR:?set LLM_GGUF_DIR to the directory containing Qwen3.5-9B-Q4_K_M.gguf}":/models:ro \
    ghcr.io/ggml-org/llama.cpp:server-cuda-b10380 -m /models/Qwen3.5-9B-Q4_K_M.gguf --alias qwen3.5-9b \
    -c 16384 -ngl 99 --host 0.0.0.0 --port 8080 --jinja -np 2
fi
if ! curl -sf localhost:18310/health >/dev/null; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${JEV_GPU:-1} \
    nohup .venv/bin/python jev_service/server.py --port 18310 > logs/jev.log 2>&1 &
  until curl -sf localhost:18310/health >/dev/null; do sleep 3; done
fi
if ! curl -sf localhost:38320/api/status >/dev/null; then
  nohup .venv/bin/python -m gomoku.app --port 38320 > logs/game.log 2>&1 &
  sleep 2
fi
echo "Game UI: http://$(hostname -I | awk '{print $1}'):38320/"
