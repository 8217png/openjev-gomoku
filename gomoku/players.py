"""Player adapters: JEV (APUS-OpenJev decision model), LLM (OpenAI-compatible chat) and algorithm bots."""

from __future__ import annotations

import ipaddress
import os
import random
import re
import time

import httpx

from .engine import (BLACK, SIZE, AlphaBetaBot, Board, describe_move, line_windows, other, parse_coord,
                     stone_lists, threat_summary, to_coord)

JEV_URL = os.environ.get("JEV_URL", "http://127.0.0.1:18310")
STARTLUX_URL = os.environ.get("STARTLUX_URL", "http://127.0.0.1:18330")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:18300/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-9b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "none")

_clients: dict[bool, httpx.Client] = {}


def _is_local(url: str) -> bool:
    host = httpx.URL(url).host
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def http_client(url: str) -> httpx.Client:
    """Local/LAN services bypass proxy env vars (httpx rejects e.g. ALL_PROXY=socks://...);
    remote URLs still honour them."""
    local = _is_local(url)
    if local not in _clients:
        try:
            _clients[local] = httpx.Client(timeout=180, trust_env=not local)
        except ValueError as exc:
            raise RuntimeError(f"proxy environment variables are unusable for {url}: {exc}") from exc
    return _clients[local]


def _stone_name(player: int) -> str:
    return "Black" if player == BLACK else "White"


def _recent(board: Board, n: int = 6) -> str:
    items = []
    for i, (r, c) in enumerate(board.moves[-n:], start=max(1, len(board.moves) - n + 1)):
        items.append(f"{i}.{'X' if i % 2 else 'O'}{to_coord(r, c)}")
    return " ".join(items) or "none"


def board_prompt(board: Board, player: int) -> str:
    """Board shown with X = Black, O = White (fixed), plus whose turn it is."""
    me = "X" if player == BLACK else "O"
    return (
        "Gomoku (freestyle: five or more in a row horizontally, vertically or diagonally wins) "
        f"on a 15x15 board. Columns A-O left to right, rows 15 (top) to 1 (bottom).\n"
        "X = Black stones, O = White stones, . = empty.\n"
        f"{board.render()}\n"
        f"Moves played: {len(board.moves)}. Recent moves: {_recent(board)}.\n"
        f"It is {_stone_name(player)}'s ({me}) turn."
    )


class Player:
    kind = "base"

    def choose(self, board: Board) -> dict:
        raise NotImplementedError

    def label(self) -> str:
        return self.kind


class AlgoPlayer(Player):
    kind = "algo"

    def __init__(self, level: str = "hard", seed: int | None = None):
        self.level = level
        self.bot = AlphaBetaBot(level, seed=seed)

    def label(self):
        return f"algo-{self.level}"

    def choose(self, board: Board) -> dict:
        res = self.bot.choose(board.copy())
        return {"move": res["move"], "meta": {k: v for k, v in res.items() if k != "move"}}


RULES = ("Priorities: 1) a move that WINS immediately; 2) otherwise BLOCK an opponent five; "
         "3) create an open four or a double threat; 4) block the opponent's open three / four points; "
         "5) otherwise prefer moves that both build your shapes and hinder the opponent.")

ATTACK_STRONG = ("creates an open four", "creates a four-three", "creates a double open three",
                 "creates a four ", "creates an open three")
# Stage-1 goals of the staged mode, in priority order: (id, description, membership test).
GOALS = (
    ("win", "WIN immediately: complete five in a row", lambda c: c["attack"].startswith("WINS")),
    ("block-five", "BLOCK the opponent's five (they win next move otherwise)",
     lambda c: c["defence"].startswith("BLOCKS")),
    ("attack", "ATTACK: make a forcing threat of your own (four / open three / double threat)",
     lambda c: c["attack"].startswith(ATTACK_STRONG)),
    ("defend", "DEFEND: block a point where the opponent makes a four or open four / open three",
     lambda c: not c["defence"].startswith(("BLOCKS", "slightly", "no defensive"))),
    ("develop", "DEVELOP: no urgent threats; build your shapes while hindering the opponent", lambda c: True),
)


class JevPlayer(Player):
    """Algorithm proposes <=16 legal candidate moves with tactical descriptions; OpenJev picks one.

    `votes` > 1 asks the model several times with differently shuffled candidate orders and averages
    the probabilities, which cancels the model's preference for particular answer letters.

    `info` selects how much the model is told (ablations for the README):
      full      board + threat summary + priority rules + attack/defence text per candidate (default)
      bare      board + candidate coordinates only
      barerules bare + the priority rules
      rich      barerules + a raw transcription of the position (no analysis): both sides' stone lists, and for
                every candidate the four lines through it (4 cells each side) — the gomoku analogue of the chess
                harness's piece lists
      staged    progressive disclosure: first pick a goal (win / block / attack / defend / develop) from
                the threat summary, then pick a move among that goal's candidates, described only by the
                aspect relevant to the goal.

    `backend` selects the decision model: "openjev" (jev_service, POST /decide) or "startlux"
    (StartLux-Decision, TypeSafe POST /v1/systemone). Both get exactly the same state, instructions and candidates.
    """

    kind = "jev"
    INFO_LEVELS = ("full", "bare", "barerules", "rich", "staged")
    BACKENDS = ("openjev", "startlux")

    def __init__(self, effort: str = "high", n_candidates: int = 10, shuffle: bool = True,
                 votes: int = 2, seed: int | None = None, url: str | None = None, info: str = "full",
                 backend: str = "openjev"):
        if info not in self.INFO_LEVELS:
            raise ValueError(f"unknown info level {info}")
        if backend not in self.BACKENDS:
            raise ValueError(f"unknown backend {backend}")
        self.backend = backend
        self.effort = effort
        self.info = info
        self.n = max(2, min(16, n_candidates))
        self.shuffle = shuffle
        self.votes = max(1, votes)
        self.rng = random.Random(seed)
        self.url = url or (STARTLUX_URL if backend == "startlux" else JEV_URL)

    def label(self):
        base = "slx" if self.backend == "startlux" else f"jev-{self.effort}"
        return base + ("" if self.info == "full" else f"-{self.info}")

    def _decide(self, req: dict) -> dict:
        """POST one choice request; returns {"probabilities", "latency_ms", ...} for either backend."""
        if self.backend == "openjev":
            resp = http_client(self.url).post(f"{self.url}/decide", json=req)
            resp.raise_for_status()
            return resp.json()
        body = {"state": req["state"], "questions": {"move": {
            "type": "choice", "instructions": req["instructions"],
            "criteria": {c["id"]: c["description"] for c in req["criteria"]}}}}  # dict order = letter order
        resp = http_client(self.url).post(f"{self.url}/v1/systemone", json=body)
        resp.raise_for_status()
        data = resp.json()
        return {"probabilities": data["answers"]["move"]["probabilities"], "latency_ms": data.get("latency_ms", 0),
                "effort": None, "executed_layers": None, "prompt_tokens": data.get("usage", {}).get("input_tokens")}

    def candidates(self, board: Board) -> list:
        player = board.to_move
        ranked = board.ranked_moves(player, self.n)
        return [describe_move(board, r, c, player) | {"rc": (r, c)} for _, r, c in ranked]

    def _request(self, state: str, instructions: str, criteria: list) -> dict:
        return {"state": state, "instructions": instructions, "criteria": criteria,
                "primitive": "choice", "effort": self.effort}

    def build_request(self, board: Board, cands: list) -> dict:
        player = board.to_move
        me = "X" if player == BLACK else "O"
        you = f"You are {_stone_name(player)} ({me}). Choose the strongest next move."
        if self.info in ("bare", "barerules"):
            return self._request(board_prompt(board, player),
                                 you + (" " + RULES if self.info == "barerules" else ""),
                                 [{"id": c["coord"], "description": f"Play {me} at {c['coord']}."} for c in cands])
        if self.info == "rich":
            return self._request(
                board_prompt(board, player) + "\n" + stone_lists(board),
                you + " " + RULES + " Each option lists the four lines through its point "
                "(X = Black, O = White, . = empty, # = off the board, [*] = the point itself).",
                [{"id": c["coord"], "description": f"Play {me} at {c['coord']}. "
                  + "; ".join(line_windows(board, *c["rc"]))} for c in cands])
        return self._request(
            board_prompt(board, player) + "\nThreat analysis: " + threat_summary(board, player),
            you + " " + RULES,
            [{"id": c["coord"], "description": f"Play {me} at {c['coord']}. Attack: {c['attack']}. "
              f"Defence: {c['defence']}."} for c in cands])

    def _vote(self, items: list, make_request) -> tuple[dict, dict, float]:
        """Ask `votes` times with shuffled `items` (each has an "id"); return averaged probs, last response, ms."""
        probs = {it["id"]: 0.0 for it in items}
        model_ms, data = 0.0, {}
        for _ in range(self.votes):
            order = items[:]
            if self.shuffle:
                # Hide the heuristic ordering so the decision comes from the model, not list position.
                self.rng.shuffle(order)
            data = self._decide(make_request(order))
            model_ms += data.get("latency_ms", 0)
            for k, v in data["probabilities"].items():
                probs[k] += v / self.votes
        return probs, data, model_ms

    def _staged(self, board: Board, cands: list) -> tuple[list, dict, dict, float]:
        """Stage 1: choose a goal. Returns (stage-2 criteria, stage-1 meta, last response, ms)."""
        player = board.to_move
        me = "X" if player == BLACK else "O"
        state = board_prompt(board, player) + "\nThreat analysis: " + threat_summary(board, player)
        groups = {}
        for gid, text, test in GOALS:
            members = [c for c in cands if test(c)]
            if members:
                groups[gid] = (text, members)
        meta, data, ms = {}, {}, 0.0
        if len(groups) > 1:
            def strongest(gid, members):
                key = "defence" if gid in ("block-five", "defend") else "attack"
                return members[0][key]  # candidates are in heuristic order, first = strongest
            goals = [{"id": gid, "description": f"{text}. {len(members)} candidate move(s); strongest effect: "
                      f"{strongest(gid, members)}."} for gid, (text, members) in groups.items()]
            probs, data, ms = self._vote(goals, lambda order: self._request(
                state, f"You are {_stone_name(player)} ({me}). Decide the GOAL of your next move. " + RULES, order))
            goal = max(probs, key=probs.get)
            meta = {"goal": goal, "goal_probs": {k: round(v, 4) for k, v in probs.items()}}
        else:
            goal = next(iter(groups))
            meta = {"goal": goal, "note": "single goal available"}
        text, members = groups[goal]
        def describe(c):
            if goal in ("win", "attack"):
                return f"Play {me} at {c['coord']}: {c['attack']}."
            if goal in ("block-five", "defend"):
                return f"Play {me} at {c['coord']}: {c['defence']}."
            return f"Play {me} at {c['coord']}. Attack: {c['attack']}. Defence: {c['defence']}."
        meta["stage2_instructions"] = f"You are {_stone_name(player)} ({me}). Goal: {text}. Choose the best move for this goal."
        return [{"id": c["coord"], "description": describe(c)} for c in members], meta, data, ms

    def choose(self, board: Board) -> dict:
        if len(board.moves) == 0:
            return {"move": (SIZE // 2, SIZE // 2), "meta": {"note": "opening at center (no candidates)"}}
        cands = self.candidates(board)
        if len(cands) == 1:
            return {"move": cands[0]["rc"], "meta": {"note": "single legal candidate"}}
        t0 = time.perf_counter()
        extra, model_ms = {}, 0.0
        if self.info == "staged":
            player = board.to_move
            criteria, extra, data, model_ms = self._staged(board, cands)
            state = board_prompt(board, player) + "\nThreat analysis: " + threat_summary(board, player)
            instr = extra.pop("stage2_instructions")
            if len(criteria) == 1:
                probs = {criteria[0]["id"]: 1.0}
            else:
                probs, data2, ms2 = self._vote(criteria, lambda order: self._request(state, instr, order))
                data, model_ms = data2, model_ms + ms2
        else:
            items = [{"id": c["coord"], "c": c} for c in cands]
            probs, data, model_ms = self._vote(
                items, lambda order: self.build_request(board, [it["c"] for it in order]))
        choice = max(probs, key=probs.get)
        rc = next(c["rc"] for c in cands if c["coord"] == choice)
        top = sorted(probs.items(), key=lambda kv: -kv[1])
        return {
            "move": rc,
            "meta": {
                "backend": self.backend,
                "effort": data.get("effort", self.effort),
                "executed_layers": data.get("executed_layers"),
                "prompt_tokens": data.get("prompt_tokens"),
                "votes": self.votes,
                "info": self.info,
                **extra,
                "model_ms": round(model_ms, 1),
                "http_ms": round((time.perf_counter() - t0) * 1000, 1),
                "top": [{"coord": k, "p": round(v, 4)} for k, v in top[:5]],
                "candidates": [{"coord": c["coord"], "tags": c["tags"]} for c in cands],
            },
        }


COORD_RE = re.compile(r"\b([A-Oa-o])\s*(1[0-5]|[1-9])\b")


class LLMPlayer(Player):
    """Free-form LLM: sees the board, must answer with a coordinate. Illegal answers are retried."""

    kind = "llm"

    def __init__(self, model: str = LLM_MODEL, base_url: str = LLM_BASE_URL, api_key: str = LLM_API_KEY,
                 thinking: bool = False, max_retries: int = 2, seed: int | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.thinking = thinking
        self.max_retries = max_retries
        self.rng = random.Random(seed)

    def label(self):
        return f"llm-{self.model}"

    def _chat(self, messages):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 4096 if self.thinking else 400,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        resp = http_client(self.base_url).post(f"{self.base_url}/chat/completions", json=body,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content") or ""

    def choose(self, board: Board) -> dict:
        player = board.to_move
        me = "X" if player == BLACK else "O"
        messages = [
            {"role": "system", "content": "You are a strong Gomoku player."},
            {"role": "user", "content": board_prompt(board, player) + (
                f"\n\nYou play {me}. Think briefly about wins, forced blocks and threats, then give your "
                "move on the last line exactly as `MOVE: <column><row>` (for example `MOVE: H8`). "
                "The square must be empty.")},
        ]
        t0 = time.perf_counter()
        errors = []
        text = ""
        for attempt in range(self.max_retries + 1):
            text = self._chat(messages)
            m = re.search(r"MOVE:\s*([A-Oa-o]\s*(?:1[0-5]|[1-9]))", text)
            found = m.group(1) if m else (COORD_RE.findall(text) or [None])[-1]
            if isinstance(found, tuple):
                found = "".join(found)
            if found:
                r, c = parse_coord(found.replace(" ", ""))
                if board.grid[r][c] == 0:
                    return {"move": (r, c), "meta": {
                        "reply": text[-600:], "attempts": attempt + 1, "errors": errors,
                        "ms": round((time.perf_counter() - t0) * 1000, 1)}}
                err = f"{found.upper()} is already occupied."
            else:
                err = "No move found. End with `MOVE: <column><row>`."
            errors.append(err)
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Invalid: {err} Choose an EMPTY square."}]
        # Fallback keeps the game going; it is recorded so illegal-move rates can be reported.
        _, r, c = self.rng.choice(board.ranked_moves(player, 5))
        return {"move": (r, c), "meta": {"reply": text[-600:], "attempts": self.max_retries + 1,
                                         "errors": errors, "fallback": True,
                                         "ms": round((time.perf_counter() - t0) * 1000, 1)}}


def make_player(spec: dict, seed: int | None = None) -> Player | None:
    """spec: {"type": "human"|"jev"|"llm"|"algo", ...options}."""
    kind = spec.get("type", "human")
    if kind == "human":
        return None
    if kind == "algo":
        return AlgoPlayer(spec.get("level", "hard"), seed=seed)
    if kind == "jev":
        return JevPlayer(spec.get("effort", "high"), int(spec.get("candidates", 10)),
                         bool(spec.get("shuffle", True)), int(spec.get("votes", 2)), seed=seed,
                         info=spec.get("info", "full"), backend=spec.get("backend", "openjev"), url=spec.get("url"))
    if kind == "llm":
        return LLMPlayer(spec.get("model") or LLM_MODEL, spec.get("base_url") or LLM_BASE_URL,
                         spec.get("api_key") or LLM_API_KEY, bool(spec.get("thinking", False)), seed=seed)
    raise ValueError(f"unknown player type {kind}")


__all__ = ["make_player", "JevPlayer", "LLMPlayer", "AlgoPlayer", "board_prompt", "other"]
