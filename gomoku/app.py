"""Gomoku web server: human / JEV / LLM / algorithm players in any combination."""

from __future__ import annotations

import argparse
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import BLACK, WHITE, Board, parse_coord, to_coord
from .players import JEV_URL, LLM_BASE_URL, http_client, make_player

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Gomoku arena")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Game:
    def __init__(self, black: dict, white: dict):
        self.id = uuid.uuid4().hex[:10]
        self.specs = {BLACK: black, WHITE: white}
        self.players = {BLACK: make_player(black), WHITE: make_player(white)}
        self.board = Board()
        self.log: list[dict] = []
        self.lock = threading.Lock()
        self.error: str | None = None

    def label(self, p):
        pl = self.players[p]
        return pl.label() if pl else "human"

    def state(self) -> dict:
        b = self.board
        return {
            "id": self.id,
            "grid": b.grid,
            "to_move": b.to_move,
            "to_move_is_ai": self.players[b.to_move] is not None and not b.over,
            "winner": b.winner,
            "draw": b.full and not b.winner,
            "win_line": b.win_line,
            "players": {"black": self.label(BLACK), "white": self.label(WHITE)},
            "specs": {"black": self.specs[BLACK], "white": self.specs[WHITE]},
            "moves": self.log,
            "error": self.error,
        }

    def record(self, player, rc, meta):
        self.board.play(*rc, player)
        self.log.append({"n": len(self.board.moves), "player": player, "who": self.label(player),
                         "rc": list(rc), "coord": to_coord(*rc), "meta": meta})


games: dict[str, Game] = {}


class NewGame(BaseModel):
    black: dict = {"type": "human"}
    white: dict = {"type": "jev", "effort": "high"}


class HumanMove(BaseModel):
    coord: str | None = None
    r: int | None = None
    c: int | None = None


def _get(gid: str) -> Game:
    if gid not in games:
        raise HTTPException(404, "no such game")
    return games[gid]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    out = {}
    for name, url in (("jev", f"{JEV_URL}/health"), ("llm", f"{LLM_BASE_URL}/models")):
        try:
            r = http_client(url).get(url, timeout=3)
            out[name] = {"ok": r.status_code == 200, "detail": r.json()}
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            out[name] = {"ok": False, "detail": str(exc)}
    return out


@app.post("/api/games")
def new_game(req: NewGame):
    try:
        g = Game(req.black, req.white)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    games[g.id] = g
    return g.state()


@app.get("/api/games/{gid}")
def get_game(gid: str):
    return _get(gid).state()


@app.post("/api/games/{gid}/move")
def human_move(gid: str, req: HumanMove):
    g = _get(gid)
    with g.lock:
        b = g.board
        if b.over:
            raise HTTPException(409, "game over")
        if g.players[b.to_move] is not None:
            raise HTTPException(409, "it is the AI's turn")
        try:
            rc = parse_coord(req.coord) if req.coord else (req.r, req.c)
            g.record(b.to_move, rc, {})
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
    return g.state()


@app.post("/api/games/{gid}/ai")
def ai_move(gid: str):
    g = _get(gid)
    if not g.lock.acquire(blocking=False):
        raise HTTPException(409, "AI is already thinking")
    try:
        b = g.board
        if b.over:
            return g.state()
        p = b.to_move
        player = g.players[p]
        if player is None:
            raise HTTPException(409, "it is the human's turn")
        try:
            res = player.choose(b.copy())
        except Exception as exc:  # noqa: BLE001 - backend failures are reported, not hidden
            g.error = f"{player.label()} failed: {exc}"
            raise HTTPException(502, g.error) from exc
        g.error = None
        g.record(p, tuple(res["move"]), res.get("meta", {}))
        return g.state()
    finally:
        g.lock.release()


@app.post("/api/games/{gid}/undo")
def undo(gid: str):
    """Take back moves until it is a human's turn again (or the board is empty)."""
    g = _get(gid)
    with g.lock:
        b = g.board
        if not b.moves:
            return g.state()
        b.undo()
        g.log.pop()
        while b.moves and g.players[b.to_move] is not None:
            b.undo()
            g.log.pop()
    return g.state()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=38320)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
