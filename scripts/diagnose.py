"""Probe JEV's candidate choices on sampled positions (no games): how often it picks the heuristic's
top move, agrees with algo-hard, and finds forced wins/blocks. Usage: python scripts/diagnose.py [n] [variant ...]"""
import collections, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gomoku.engine import AlphaBetaBot, Board
from gomoku.players import JevPlayer

rng = random.Random(1)
positions = []
while len(positions) < (int(sys.argv[1]) if len(sys.argv) > 1 else 60):
    b = Board(); bots = [AlphaBetaBot("easy", seed=rng.randint(0, 9999)) for _ in range(2)]
    n = rng.randint(6, 30)
    while not b.over and len(b.moves) < n:
        b.play(*bots[len(b.moves) % 2].choose(b)["move"])
    if not b.over:
        positions.append(b)
hard = [AlphaBetaBot("hard").choose(b.copy())["move"] for b in positions]

VARIANTS = {"high n8 v1": dict(n_candidates=8, votes=1), "high n8 v2": dict(n_candidates=8, votes=2),
            "high n10 v2": dict(n_candidates=10, votes=2),
            "high n10 v2 bare": dict(n_candidates=10, votes=2, info="bare"),
            "high n10 v2 barerules": dict(n_candidates=10, votes=2, info="barerules"),
            "high n10 v2 staged": dict(n_candidates=10, votes=2, info="staged"),
            "high n10 v2 rich": dict(n_candidates=10, votes=2, info="rich"), "high n5 v2": dict(n_candidates=5, votes=2),
            "low n8 v2": dict(effort="low", n_candidates=8, votes=2),
            **{f"slx {i}": dict(n_candidates=10, votes=2, info=i, backend="startlux")
               for i in ("full", "bare", "barerules", "rich", "staged")},
            **{f"slx27 {i}": dict(n_candidates=10, votes=2, info=i, backend="startlux",
                                  url=os.environ.get("STARTLUX27_URL", "http://127.0.0.1:18332"))
               for i in ("full", "bare", "barerules", "rich", "staged")}}
only = sys.argv[2:]  # optional variant names to run, e.g. "high n10 v2" "high n10 v2 bare"
for name, kw in VARIANTS.items():
    if only and name not in only:
        continue
    jev = JevPlayer(seed=7, **kw)
    ranks, agree, forced = collections.Counter(), 0, [0, 0]
    for b, h in zip(positions, hard):
        cands = jev.candidates(b)
        move = tuple(jev.choose(b)["move"])
        ranks[[c["rc"] for c in cands].index(move)] += 1
        agree += move == tuple(h)
        must = [c["rc"] for c in cands if c["attack"].startswith("WINS") or c["defence"].startswith("BLOCKS")]
        if must:
            forced[1] += 1; forced[0] += move in must
    top1 = ranks[0] / len(positions)
    print(f"{name:<16} top1={top1:.0%} agree_hard={agree / len(positions):.0%} forced={forced[0]}/{forced[1]} "
          f"rank_dist={dict(sorted(ranks.items()))}", flush=True)
