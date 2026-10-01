"""Gomoku (freestyle, 15x15, five-or-more wins) board, pattern evaluation and alpha-beta bot."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from functools import lru_cache

SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2
COLS = "ABCDEFGHIJKLMNO"
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
WIN_SCORE = 10_000_000


def other(player: int) -> int:
    return BLACK if player == WHITE else WHITE


def to_coord(r: int, c: int) -> str:
    """Row 0 is the top line and is printed as 15; column 0 is A."""
    return f"{COLS[c]}{SIZE - r}"


def parse_coord(text: str) -> tuple[int, int]:
    text = text.strip().upper()
    c = COLS.index(text[0])
    n = int(text[1:])
    if not 1 <= n <= SIZE:
        raise ValueError(f"bad coordinate {text}")
    return SIZE - n, c


# ---------------------------------------------------------------- patterns
# Line strings are from one player's perspective: 1 = own stone, 0 = empty, 2 = blocked.
# Scores are counted per occurrence of the pattern in the (padded) line.
PATTERNS = (
    ("11111", 1_000_000, "five"),
    ("011110", 100_000, "open_four"),
    ("011112", 10_000, "four"),
    ("211110", 10_000, "four"),
    ("10111", 10_000, "four"),
    ("11011", 10_000, "four"),
    ("11101", 10_000, "four"),
    ("001110", 3_000, "open_three"),
    ("011100", 3_000, "open_three"),
    ("010110", 3_000, "open_three"),
    ("011010", 3_000, "open_three"),
    ("001112", 500, "three"),
    ("211100", 500, "three"),
    ("010112", 500, "three"),
    ("211010", 500, "three"),
    ("011012", 500, "three"),
    ("210110", 500, "three"),
    ("10011", 500, "three"),
    ("11001", 500, "three"),
    ("10101", 500, "three"),
    ("001100", 200, "open_two"),
    ("001010", 200, "open_two"),
    ("010100", 200, "open_two"),
    ("010010", 100, "open_two"),
    ("000112", 30, "two"),
    ("211000", 30, "two"),
    ("001012", 30, "two"),
    ("210100", 30, "two"),
    ("010012", 30, "two"),
    ("210010", 30, "two"),
)


def _count(s: str, pat: str) -> int:
    n, i = 0, s.find(pat)
    while i != -1:
        n += 1
        i = s.find(pat, i + 1)
    return n


@lru_cache(maxsize=500_000)
def line_score(s: str) -> int:
    s = "2" + s + "2"
    if "11111" in s:
        return 1_000_000
    total = 0
    for pat, val, _ in PATTERNS[1:]:
        if pat in s:
            total += val * _count(s, pat)
    return total


@lru_cache(maxsize=500_000)
def line_shapes(s: str) -> frozenset:
    s = "2" + s + "2"
    return frozenset(name for pat, _, name in PATTERNS if pat in s)


def _build_lines():
    lines = []
    for r in range(SIZE):
        lines.append([(r, c) for c in range(SIZE)])
    for c in range(SIZE):
        lines.append([(r, c) for r in range(SIZE)])
    for k in range(-(SIZE - 5), SIZE - 4):
        lines.append([(r, r - k) for r in range(SIZE) if 0 <= r - k < SIZE])
    for k in range(4, 2 * SIZE - 5):
        lines.append([(r, k - r) for r in range(SIZE) if 0 <= k - r < SIZE])
    cell_lines = {(r, c): [] for r in range(SIZE) for c in range(SIZE)}
    for li, cells in enumerate(lines):
        for pos, cell in enumerate(cells):
            cell_lines[cell].append((li, pos))
    return lines, cell_lines


LINES, CELL_LINES = _build_lines()


def _line_str(board, cells, player: int) -> str:
    opp = other(player)
    return "".join(
        "1" if board[r][c] == player else "2" if board[r][c] == opp else "0"
        for r, c in cells
    )


# ---------------------------------------------------------------- board
@dataclass
class Board:
    grid: list = field(default_factory=lambda: [[EMPTY] * SIZE for _ in range(SIZE)])
    moves: list = field(default_factory=list)
    winner: int = EMPTY
    win_line: list = field(default_factory=list)

    def __post_init__(self):
        # Cached per-line score for each player: score[player][line_index]
        self.score = {BLACK: [0] * len(LINES), WHITE: [0] * len(LINES)}
        self.total = {BLACK: 0, WHITE: 0}
        for li in range(len(LINES)):
            self._rescore_line(li)

    @property
    def to_move(self) -> int:
        return BLACK if len(self.moves) % 2 == 0 else WHITE

    @property
    def full(self) -> bool:
        return len(self.moves) >= SIZE * SIZE

    @property
    def over(self) -> bool:
        return self.winner != EMPTY or self.full

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.grid = [row[:] for row in self.grid]
        b.moves = self.moves[:]
        b.winner = self.winner
        b.win_line = self.win_line[:]
        b.score = {p: v[:] for p, v in self.score.items()}
        b.total = dict(self.total)
        return b

    def _rescore_line(self, li: int):
        cells = LINES[li]
        for p in (BLACK, WHITE):
            new = line_score(_line_str(self.grid, cells, p))
            self.total[p] += new - self.score[p][li]
            self.score[p][li] = new

    def play(self, r: int, c: int, player: int | None = None, check_win: bool = True):
        if self.winner:
            raise ValueError("game is over")
        if not (0 <= r < SIZE and 0 <= c < SIZE) or self.grid[r][c] != EMPTY:
            raise ValueError(f"illegal move {r},{c}")
        player = player or self.to_move
        if player != self.to_move:
            raise ValueError("not this player's turn")
        self.grid[r][c] = player
        self.moves.append((r, c))
        for li, _ in CELL_LINES[(r, c)]:
            self._rescore_line(li)
        if check_win:
            line = self.five_through(r, c)
            if line:
                self.winner, self.win_line = player, line

    def undo(self):
        r, c = self.moves.pop()
        self.grid[r][c] = EMPTY
        self.winner, self.win_line = EMPTY, []
        for li, _ in CELL_LINES[(r, c)]:
            self._rescore_line(li)

    def five_through(self, r: int, c: int) -> list:
        p = self.grid[r][c]
        for dr, dc in DIRS:
            line = [(r, c)]
            for sign in (1, -1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < SIZE and 0 <= cc < SIZE and self.grid[rr][cc] == p:
                    line.append((rr, cc))
                    rr, cc = rr + sign * dr, cc + sign * dc
            if len(line) >= 5:
                return sorted(line)
        return []

    def evaluate(self, player: int) -> int:
        """Static evaluation from `player`'s view; the side to move gets a tempo bonus."""
        opp = other(player)
        mine, theirs = self.total[player], self.total[opp]
        if self.to_move == player:
            return mine - theirs * 0.9
        return mine * 0.9 - theirs

    # ------------------------------------------------------------ move analysis
    def neighbors(self, dist: int = 2) -> list:
        if not self.moves:
            return [(SIZE // 2, SIZE // 2)]
        seen = set()
        for r, c in self.moves:
            for dr in range(-dist, dist + 1):
                for dc in range(-dist, dist + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < SIZE and 0 <= cc < SIZE and self.grid[rr][cc] == EMPTY:
                        seen.add((rr, cc))
        return list(seen)

    def point_gain(self, r: int, c: int, player: int) -> int:
        """How much `player`'s line score rises by placing a stone at (r, c)."""
        before = after = 0
        self.grid[r][c] = player
        for li, _ in CELL_LINES[(r, c)]:
            after += line_score(_line_str(self.grid, LINES[li], player))
            before += self.score[player][li]
        self.grid[r][c] = EMPTY
        return after - before

    def shapes_at(self, r: int, c: int, player: int) -> list:
        """Shapes (per direction) that `player` would newly form through (r, c) by playing there."""
        out = []
        for li, pos in CELL_LINES[(r, c)]:
            cells = LINES[li]
            window = cells[max(0, pos - 5): pos + 6]
            before = line_shapes(_line_str(self.grid, window, player))
            self.grid[r][c] = player
            after = line_shapes(_line_str(self.grid, window, player))
            self.grid[r][c] = EMPTY
            out.append(after - before)
        return out

    def ranked_moves(self, player: int, limit: int | None = None) -> list:
        """Candidate moves ordered by attack + defence heuristic: [(score, r, c)]."""
        opp = other(player)
        scored = []
        for r, c in self.neighbors():
            attack = self.point_gain(r, c, player)
            defend = self.point_gain(r, c, opp)
            scored.append((attack * 1.1 + defend, r, c))
        scored.sort(reverse=True)
        return scored[:limit] if limit else scored

    def winning_moves(self, player: int) -> list:
        wins = []
        for r, c in self.neighbors(1):
            self.grid[r][c] = player
            if self.five_through(r, c):
                wins.append((r, c))
            self.grid[r][c] = EMPTY
        return wins

    def render(self, x_player: int = BLACK, highlight=None) -> str:
        """ASCII board: X = x_player stones, O = the other side, . = empty."""
        sym = {EMPTY: ".", x_player: "X", other(x_player): "O"}
        rows = ["    " + " ".join(COLS)]
        for r in range(SIZE):
            cells = []
            for c in range(SIZE):
                ch = sym[self.grid[r][c]]
                cells.append(ch)
            rows.append(f"{SIZE - r:>2}  " + " ".join(cells))
        return "\n".join(rows)


# ---------------------------------------------------------------- tactical description
SHAPE_TEXT = {
    "five": "five in a row",
    "open_four": "open four",
    "four": "four",
    "open_three": "open three",
    "three": "closed three",
    "open_two": "open two",
}


def _best_shapes(shape_sets) -> list:
    order = ["five", "open_four", "four", "open_three", "three", "open_two"]
    found = []
    for s in shape_sets:
        for name in order:
            if name in s:
                found.append(name)
                break
    return sorted(found, key=order.index)


def _attack_text(own: list) -> str:
    fours = sum(1 for s in own if s in ("open_four", "four"))
    threes = own.count("open_three")
    if "five" in own:
        return "WINS immediately: completes five in a row"
    if "open_four" in own or fours >= 2:
        return "creates an open four / double four (unstoppable, wins next move)"
    if fours and threes:
        return "creates a four-three double threat (usually winning)"
    if threes >= 2:
        return "creates a double open three (strong double threat)"
    if fours:
        return "creates a four (opponent is forced to block)"
    if threes:
        return "creates an open three (threatens an open four next move)"
    if "three" in own:
        return "makes a closed three (weak)"
    if own.count("open_two") >= 2:
        return "builds two open twos (good development)"
    if "open_two" in own:
        return "builds an open two"
    return "no new shape"


def _defence_text(block: list) -> str:
    fours = sum(1 for s in block if s in ("open_four", "four"))
    if "five" in block:
        return "BLOCKS the opponent's five (must-block, otherwise we lose)"
    if "open_four" in block or fours >= 2:
        return "blocks the opponent's open-four point (answers their open three)"
    if fours and "open_three" in block:
        return "blocks the opponent's four-three point"
    if fours:
        return "blocks a point where the opponent could make a four"
    if block.count("open_three") >= 2:
        return "blocks the opponent's double-open-three point"
    if "open_three" in block:
        return "blocks a point where the opponent could make an open three"
    if "open_two" in block:
        return "slightly hinders the opponent (their open-two point)"
    return "no defensive effect"


def describe_move(board: Board, r: int, c: int, player: int) -> dict:
    """Tactical facts about playing (r, c) for `player`: what it builds and what it blocks."""
    opp = other(player)
    own = _best_shapes(board.shapes_at(r, c, player))
    block = _best_shapes(board.shapes_at(r, c, opp))
    attack, defence = _attack_text(own), _defence_text(block)
    dist = max(abs(r - SIZE // 2), abs(c - SIZE // 2))
    return {"coord": to_coord(r, c), "own": own, "blocks": block, "attack": attack,
            "defence": defence, "tags": [f"attack: {attack}", f"defence: {defence}"], "center_dist": dist}


def threat_summary(board: Board, player: int) -> str:
    """Plain-language list of urgent points for both sides, used as decision context."""
    opp = other(player)
    lines = []
    my_win = [to_coord(*m) for m in board.winning_moves(player)]
    opp_win = [to_coord(*m) for m in board.winning_moves(opp)]
    my_o4, opp_o4, opp_f4 = [], [], []
    for r, c in board.neighbors():
        mine = _best_shapes(board.shapes_at(r, c, player))
        theirs = _best_shapes(board.shapes_at(r, c, opp))
        if "open_four" in mine:
            my_o4.append(to_coord(r, c))
        if "open_four" in theirs:
            opp_o4.append(to_coord(r, c))
        elif "four" in theirs and "open_three" in theirs:
            opp_f4.append(to_coord(r, c))
    if my_win:
        lines.append(f"You can WIN right now at: {', '.join(my_win)}.")
    if opp_win:
        lines.append(f"DANGER: opponent completes five next move at: {', '.join(opp_win)} — you must block.")
    if my_o4:
        lines.append(f"You can make an open four at: {', '.join(sorted(my_o4))}.")
    if opp_o4:
        lines.append(f"Opponent has an open three: they make an open four at {', '.join(sorted(opp_o4))} "
                     "unless you block one of those points (or you have a stronger forcing move).")
    if opp_f4:
        lines.append(f"Opponent four-three threat points: {', '.join(sorted(opp_f4))}.")
    return " ".join(lines) or "No immediate threats for either side."


# ---------------------------------------------------------------- alpha-beta bot
LEVELS = {
    "easy": {"depth": 1, "width": 8, "noise": 0.25},
    "medium": {"depth": 2, "width": 10, "noise": 0.0},
    "hard": {"depth": 4, "width": 10, "noise": 0.0},
    "expert": {"depth": 6, "width": 12, "noise": 0.0},
}


class SearchTimeout(Exception):
    pass


class AlphaBetaBot:
    def __init__(self, level: str = "medium", time_limit: float = 5.0, seed: int | None = None):
        if level not in LEVELS:
            raise ValueError(f"unknown level {level}")
        self.level = level
        self.cfg = LEVELS[level]
        self.time_limit = time_limit
        self.rng = random.Random(seed)
        self.nodes = 0

    def _forced(self, board: Board, player: int):
        wins = board.winning_moves(player)
        if wins:
            return wins[0], "win"
        blocks = board.winning_moves(other(player))
        if blocks:
            return blocks[0], "block"
        return None, None

    def choose(self, board: Board) -> dict:
        t0 = time.time()
        player = board.to_move
        self.nodes = 0
        if not board.moves:
            return {"move": (SIZE // 2, SIZE // 2), "reason": "opening", "ms": 0}
        move, reason = self._forced(board, player)
        if move:
            return {"move": move, "reason": reason, "ms": int((time.time() - t0) * 1000)}
        cfg = self.cfg
        ranked = board.ranked_moves(player, cfg["width"])
        if cfg["noise"]:
            top = ranked[0][0]
            pool = [m for m in ranked if m[0] >= top * (1 - cfg["noise"])]
            _, r, c = self.rng.choice(pool)
            return {"move": (r, c), "reason": "heuristic", "ms": int((time.time() - t0) * 1000)}
        deadline = t0 + self.time_limit
        best = (ranked[0][1], ranked[0][2])
        best_val = None
        # Iterative deepening so a timeout still returns the last completed depth.
        for depth in range(1, cfg["depth"] + 1):
            try:
                val, mv = self._root(board, player, depth, deadline, ranked)
                best, best_val = mv, val
            except SearchTimeout:
                break
            if val >= WIN_SCORE // 2:
                break
        return {
            "move": best,
            "reason": "search",
            "value": best_val,
            "nodes": self.nodes,
            "ms": int((time.time() - t0) * 1000),
        }

    def _root(self, board, player, depth, deadline, ranked):
        alpha, beta = -float("inf"), float("inf")
        best = None
        for _, r, c in ranked:
            board.play(r, c, player)
            try:
                if board.winner:
                    val = WIN_SCORE
                else:
                    val = -self._negamax(board, other(player), depth - 1, -beta, -alpha, deadline)
            finally:
                board.undo()
            if best is None or val > alpha:
                alpha, best = val, (r, c)
        return alpha, best

    def _negamax(self, board, player, depth, alpha, beta, deadline):
        self.nodes += 1
        if self.nodes & 255 == 0 and time.time() > deadline:
            raise SearchTimeout
        if depth <= 0:
            return board.evaluate(player)
        wins = board.winning_moves(player)
        if wins:
            return WIN_SCORE - (10 - depth)
        threats = board.winning_moves(other(player))
        if len(threats) >= 2:
            return -WIN_SCORE + (10 - depth)
        if threats:
            moves = [(0, threats[0][0], threats[0][1])]
        else:
            moves = board.ranked_moves(player, self.cfg["width"])
        if not moves:
            return 0
        best = -float("inf")
        for _, r, c in moves:
            board.play(r, c, player, check_win=False)
            try:
                val = -self._negamax(board, other(player), depth - 1, -beta, -alpha, deadline)
            finally:
                board.undo()
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best
