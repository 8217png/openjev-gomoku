#!/usr/bin/env bash
# Sequential JEV matches (the reference runtime serves one request at a time).
cd "$(dirname "$0")/.."
P=.venv/bin/python
for opp in easy medium hard; do $P scripts/arena.py jev-high algo-$opp --games 20 > logs/arena_jev-high_$opp.log 2>&1; done
$P scripts/arena.py jev-low algo-easy --games 20 > logs/arena_jev-low_easy.log 2>&1
$P scripts/arena.py jev-high llm --games 6 > logs/arena_jev-high_llm.log 2>&1
