import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gomoku.engine import BLACK, WHITE, AlphaBetaBot, Board, describe_move, parse_coord, to_coord  # noqa: E402


def setup(black, white):
    """Build a position from coordinate lists, alternating colours (padding with far-away moves)."""
    b = Board()
    spare = iter([(r, c) for r in (0, 14) for c in range(15)])
    bl, wh = [parse_coord(x) for x in black], [parse_coord(x) for x in white]
    while bl or wh:
        if b.to_move == BLACK:
            b.play(*(bl.pop(0) if bl else next(spare)))
        else:
            b.play(*(wh.pop(0) if wh else next(spare)))
    return b


def test_coords_roundtrip():
    for r in range(15):
        for c in range(15):
            assert parse_coord(to_coord(r, c)) == (r, c)
    assert to_coord(7, 7) == "H8"


@pytest.mark.parametrize("line", [
    ["D4", "E5", "F6", "G7", "H8"],       # diagonal
    ["C10", "D10", "E10", "F10", "G10"],  # row
    ["K1", "K2", "K3", "K4", "K5"],       # column at edge
    ["L4", "K5", "J6", "I7", "H8"],       # anti-diagonal
])
def test_five_detected(line):
    b = setup(line, ["A15", "B15", "C15", "D15"])
    assert b.winner == BLACK
    assert len(b.win_line) == 5


def test_undo_restores_scores():
    b = setup(["H8", "H9"], ["J8"])
    before = (dict(b.total), [row[:] for row in b.grid])
    b.play(*parse_coord("H10"))
    b.undo()
    assert (dict(b.total), b.grid) == before


@pytest.mark.parametrize("level", ["easy", "medium", "hard", "expert"])
def test_bot_takes_win_and_blocks(level):
    win = setup(["H8", "H9", "H10", "H11"], ["J8", "J9", "J10"])
    assert to_coord(*AlphaBetaBot(level).choose(win)["move"]) in ("H7", "H12")
    block = setup(["A1", "C1", "E1"], ["H8", "H9", "H10", "H11"])
    assert to_coord(*AlphaBetaBot(level).choose(block)["move"]) in ("H7", "H12")


def test_describe_move_tags():
    b = setup(["H8", "H9", "H10", "H11"], ["J8", "J9", "J10"])
    assert "WINS" in describe_move(b, *parse_coord("H12"), BLACK)["tags"][0]
    b2 = setup(["A1", "C1", "E1", "G1"], ["H8", "H9", "H10", "H11"])
    assert any("BLOCKS" in t for t in describe_move(b2, *parse_coord("H7"), BLACK)["tags"])
    b3 = setup(["H8", "H9"], ["A1"])
    assert "open three" in " ".join(describe_move(b3, *parse_coord("H10"), BLACK)["tags"])


def test_search_timeout_leaves_board_intact():
    b = setup(["H8", "I9", "G7"], ["H9", "I8"])
    snapshot = [row[:] for row in b.grid]
    AlphaBetaBot("expert", time_limit=0.01).choose(b)
    assert b.grid == snapshot and b.to_move == WHITE
