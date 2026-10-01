"""Batch matches between players; colours alternate and each random opening is played twice.

Examples:
  python scripts/arena.py jev-high algo-medium --games 20
  python scripts/arena.py jev-high llm --games 4
Player specs: human is not allowed; jev-high | jev-low | llm | llm-think | algo-<easy|medium|hard|expert>
             | cand-top1 | cand-random (controls: pick from the same candidate list JEV sees).
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gomoku.engine import BLACK, SIZE, WHITE, Board, to_coord  # noqa: E402
from gomoku.players import Player, make_player  # noqa: E402


class CandidateControl(Player):
    """Same candidate generator as JEV, but chooses top-1 or uniformly at random (no model)."""

    def __init__(self, mode, n=10, seed=None):
        self.mode, self.n, self.rng = mode, n, random.Random(seed)
        self.kind = f"cand-{mode}"

    def label(self):
        return self.kind

    def choose(self, board):
        ranked = board.ranked_moves(board.to_move, self.n)
        _, r, c = ranked[0] if self.mode == "top1" else self.rng.choice(ranked)
        return {"move": (r, c), "meta": {}}


def build(name, seed):
    if name.startswith("cand-"):
        return CandidateControl(name[5:], seed=seed)
    if name.startswith("jev"):
        return make_player({"type": "jev", "effort": name.split("-")[1] if "-" in name else "high"}, seed)
    if name.startswith("algo"):
        return make_player({"type": "algo", "level": name.split("-")[1]}, seed)
    if name in ("llm", "llm-think"):
        return make_player({"type": "llm", "thinking": name == "llm-think"}, seed)
    raise SystemExit(f"unknown player {name}")


def random_opening(rng, stones=3):
    """Black at centre, then random stones within distance 2 (a common arena opening scheme)."""
    mid = SIZE // 2
    moves = [(mid, mid)]
    while len(moves) < stones:
        m = (mid + rng.randint(-2, 2), mid + rng.randint(-2, 2))
        if m not in moves:
            moves.append(m)
    return moves


def play(black, white, opening, max_moves=SIZE * SIZE):
    b = Board()
    for rc in opening:
        b.play(*rc)
    players = {BLACK: black, WHITE: white}
    stats = {BLACK: Counter(), WHITE: Counter()}
    record = []
    while not b.over and len(b.moves) < max_moves:
        p = b.to_move
        t0 = time.perf_counter()
        res = players[p].choose(b.copy())
        stats[p]["ms"] += (time.perf_counter() - t0) * 1000
        stats[p]["moves"] += 1
        meta = res.get("meta", {})
        stats[p]["fallback"] += int(bool(meta.get("fallback")))
        stats[p]["retries"] += max(0, meta.get("attempts", 1) - 1)
        # Missed tactics: a win was available but not taken / a must-block was ignored.
        wins, threats = b.winning_moves(p), b.winning_moves(3 - p)
        if wins and tuple(res["move"]) not in wins:
            stats[p]["missed_win"] += 1
        elif threats and not wins and tuple(res["move"]) not in threats:
            stats[p]["missed_block"] += 1
        b.play(*res["move"])
        record.append(to_coord(*res["move"]))
    return b.winner, len(b.moves), record, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=10, help="even number; each opening played with both colours")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    score = Counter()
    agg = {args.a: Counter(), args.b: Counter()}
    games = []
    for g in range(args.games):
        if g % 2 == 0:
            opening = random_opening(rng)
        swap = g % 2 == 1
        names = (args.b, args.a) if swap else (args.a, args.b)
        black, white = build(names[0], args.seed * 1000 + g), build(names[1], args.seed * 1000 + g + 500)
        t0 = time.time()
        winner, n, record, stats = play(black, white, opening)
        win_name = names[0] if winner == BLACK else names[1] if winner == WHITE else "draw"
        score[win_name] += 1
        for p, nm in ((BLACK, names[0]), (WHITE, names[1])):
            agg[nm].update(stats[p])
        games.append({"black": names[0], "white": names[1], "winner": win_name, "moves": n,
                      "opening": [to_coord(*m) for m in opening], "record": record})
        print(f"[{g + 1}/{args.games}] B={names[0]:<12} W={names[1]:<12} -> {win_name:<12} "
              f"{n:>3} moves {time.time() - t0:6.1f}s", flush=True)
    summary = {"a": args.a, "b": args.b, "games": args.games, "score": dict(score), "per_player": {}}
    for nm, st in agg.items():
        summary["per_player"][nm] = {
            "wins": score[nm], "avg_ms_per_move": round(st["ms"] / max(1, st["moves"]), 1),
            "moves": st["moves"], "missed_win": st["missed_win"], "missed_block": st["missed_block"],
            "llm_retries": st["retries"], "llm_fallbacks": st["fallback"]}
    print(json.dumps(summary, indent=1))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    path = out / f"{args.a}_vs_{args.b}_s{args.seed}.json"
    path.write_text(json.dumps({"summary": summary, "games": games}, indent=1, ensure_ascii=False))
    print("saved", path)


if __name__ == "__main__":
    main()
