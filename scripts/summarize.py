"""Print a markdown table of all arena result files in a directory."""
import json, sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
print("| A | B | 局数 | A胜 | B胜 | 平 | A漏堵 | B漏堵 | A 平均ms/步 | B 平均ms/步 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for f in sorted(d.glob("*.json")):
    s = json.load(open(f))["summary"]
    a, b, pa, pb = s["a"], s["b"], s["per_player"][s["a"]], s["per_player"][s["b"]]
    print(f"| {a} | {b} | {s['games']} | {pa['wins']} | {pb['wins']} | {s['score'].get('draw', 0)} | "
          f"{pa['missed_block']} | {pb['missed_block']} | {pa['avg_ms_per_move']} | {pb['avg_ms_per_move']} |")
