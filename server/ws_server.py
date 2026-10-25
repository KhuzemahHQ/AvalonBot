"""
FastAPI WebSocket server for the Avalon spectator frontend.

Modes:
- Replay: steps through a recorded arena game (`games/game_*.jsonl`), rebuilding every player's
  memory and knowledge graph as it goes (with `talk_edges.json` from the same folder, if present).
- Live: runs a heuristic game, or an LLM game whose model calls share one queue.

Speech: each utterance is synthesized with Higgs TTS. A live game with images in `avatars/`
renders a talking clip of that seat's face. Otherwise a running LiveKit avatar speaks, or the
browser plays the WAV.

Messages to the browser (all {"type", "seq", "data"}):
  game_list, game_start, role_reveal, utterance, evil_chat, speaking, audio, clip_open,
  clip_chunk, clip_end, team_proposed,
  team_vote_result, quest_result, assassination, game_over, memory_update, replay_end, error

Run:  py -3.12 -m uvicorn server.ws_server:app --port 8000
"""

import asyncio
import base64
import contextlib
import glob
import io
import json
import os
import re
import sys
import uuid
import wave
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from avalon.agents import HeuristicAgent  # noqa: E402
from avalon.llm_agent import LLMAgent  # noqa: E402
from avalon.arena_adapter import ArenaGraphAdapter  # noqa: E402
from avalon.engine import ROLE_NAMES, Role  # noqa: E402
from avalon.replay import load_events, load_talk_edges  # noqa: E402
from avalon.runner import AvalonGameRunner  # noqa: E402
from server.audio_proxy import BOSON_API_KEY, get_tts_audio  # noqa: E402
from server.face_clips import face_stamp, iter_clip_chunks, still_for_seat  # noqa: E402
from server.livekit_avatars import AvatarManager  # noqa: E402
from server.memory_view import serialize_memories  # noqa: E402
from server.human_web_agent import HumanWebAgent  # noqa: E402

GAME_PATTERNS = ["runs/**/games/game_*.jsonl", "experiments/**/games/game_*.jsonl"]

avatar_manager = AvatarManager()


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await avatar_manager.detach_voices()


app = FastAPI(title="AvalonBot Spectator Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def list_game_files() -> List[str]:
    found = set()
    for pattern in GAME_PATTERNS:
        for path in glob.glob(os.path.join(ROOT, pattern), recursive=True):
            if ".partial." not in path:
                found.add(os.path.relpath(path, ROOT).replace("\\", "/"))
    return sorted(found)


def talk_edges_for(game_path: str) -> List[Dict[str, Any]]:
    """Hand-labelled or extracted talk edges stored next to the game's `games/` folder."""
    folder = os.path.dirname(os.path.dirname(game_path))
    path = os.path.join(folder, "talk_edges.json")
    return load_talk_edges(path) if os.path.exists(path) else []


def wav_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes() / float(w.getframerate())


@app.get("/api/health")
async def health():
    return {"status": "ok", "has_boson_key": bool(BOSON_API_KEY), "avatars": avatar_manager.is_available}


@app.get("/api/games")
async def api_games():
    return JSONResponse({"games": list_game_files()})


@app.get("/api/avatar/status")
async def avatar_status():
    return JSONResponse(avatar_manager.get_room_info())


@app.get("/api/face/{seat}")
async def face_still(seat: int):
    """Portrait for one seat. The URL is the seat number, not the file's name."""
    still = still_for_seat(seat)
    if still is None:
        return JSONResponse({"error": "no face"}, status_code=404)
    body, mime = still
    return Response(content=body, media_type=mime, headers={"Cache-Control": "no-store"})


@app.get("/api/faces/stamp")
async def faces_stamp():
    """A token the portraits use so a replaced image shows up without a hard refresh."""
    from server.face_clips import faces
    faces()
    return JSONResponse({"stamp": face_stamp()})


@app.post("/api/avatar/start")
async def avatar_start():
    if not avatar_manager.is_available:
        return JSONResponse({"error": "LiveKit or Boson credentials not configured", "available": False}, status_code=503)
    await avatar_manager.start()
    return JSONResponse(avatar_manager.get_room_info())


@app.post("/api/avatar/say/{seat}")
async def avatar_say(seat: int, text: str = "I am a loyal servant of Arthur."):
    """Makes one seat's avatar speak a line, to check voice and lip-sync."""
    wav_b64 = await get_tts_audio(text, seat)
    if not wav_b64:
        return JSONResponse({"error": "TTS failed"}, status_code=502)
    spoke = await avatar_manager.speak(seat, base64.b64decode(wav_b64))
    return JSONResponse({"spoke": spoke, "seconds": round(wav_seconds(base64.b64decode(wav_b64)), 2)})


@app.delete("/api/avatar/release")
async def avatar_release():
    await avatar_manager.release_all()
    return JSONResponse({"released": True})


@app.get("/api/avatar/token")
async def avatar_token():
    if not avatar_manager.is_available:
        return JSONResponse({"error": "LiveKit not configured"}, status_code=503)
    return JSONResponse({
        "token": avatar_manager.get_browser_token(f"spectator-{uuid.uuid4().hex[:8]}"),
        "url": os.getenv("LIVEKIT_URL", ""),
        "room": avatar_manager.room_name,
        "identities": avatar_manager.identities(),
    })


SEAT_NAMES = ("Atlas", "Lyra", "Griffin", "Phoenix", "Solara")


def with_player_names(text: str) -> str:
    """Player 0 / P0 become Atlas, and the same for the other seats. Safe to run twice."""
    if not text:
        return text

    def repl(match: re.Match) -> str:
        seat = int(match.group(1))
        return SEAT_NAMES[seat] if 0 <= seat < len(SEAT_NAMES) else match.group(0)

    text = re.sub(r"\b[Pp]layers?\s+([0-4])\b", repl, text)
    return re.sub(r"\b[Pp]([0-4])\b", repl, text)


# ── Session: one spectator connection ────────────────────────────────

class GameSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.paused = False
        self.speed = 1.0
        self.voice = True
        self.wake = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.closed = False
        self.human_seat: Optional[int] = None
        self.human_agent: Optional[HumanWebAgent] = None
        self.face_clips = False
        # Players already have each line. This queue is the spectator clock:
        # a transcript line appears when its audio starts, and later events wait
        # behind the lines still being spoken.
        self.speech_epoch = 0
        self._speech_pending = 0
        self._speech_drained = asyncio.Event()
        self._speech_drained.set()
        self._speech: asyncio.Queue = asyncio.Queue()
        self._speech_worker = asyncio.create_task(self._play_speech())
        self._clip_wait: Optional[tuple] = None

    def resolve_human_action(self, action: str, data: Dict[str, Any]) -> None:
        if self.human_agent:
            self.human_agent.resolve_action(action, data)

    async def send(self, type_: str, data: Dict[str, Any], seq: int = 0) -> None:
        if self.closed:
            return
        try:
            await self.ws.send_json({"type": type_, "seq": seq, "data": data})
        except Exception:
            self.closed = True

    def start(self, coro) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.drop_pending_speech()
        self.paused = False
        self.task = asyncio.create_task(self._run(coro))

    async def _run(self, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await self.send("error", {"message": f"{type(exc).__name__}: {exc}"})
            raise

    def step(self) -> None:
        self.wake.set()

    async def gate(self) -> None:
        """Blocks while paused, until the spectator resumes or steps once."""
        while self.paused:
            self.wake.clear()
            await self.wake.wait()
            if self.paused:  # a single step
                return

    async def sleep(self, seconds: float) -> None:
        """Waits `seconds` (scaled by speed); a step or resume cuts it short."""
        self.wake.clear()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.wake.wait(), seconds / max(self.speed, 0.1))

    def drop_pending_speech(self) -> None:
        """Drops lines not yet playing. The line already in the mouth finishes."""
        self.speech_epoch += 1
        self._release_clip_wait()
        dropped = 0
        while True:
            try:
                item = self._speech.get_nowait()
            except asyncio.QueueEmpty:
                break
            dropped += 1
            if item[0] == "audio":
                prep = item[5]
                if not prep.done():
                    prep.cancel()
        self._speech_pending -= dropped
        if self._speech_pending <= 0:
            self._speech_pending = 0
            self._speech_drained.set()

    def _enqueue(self, item: tuple) -> None:
        self._speech_pending += 1
        self._speech_drained.clear()
        self._speech.put_nowait(item)

    def enqueue_speech(self, seat: int, text: str, seq: int, reveal: Optional[tuple] = None) -> None:
        """Starts synthesis now. The transcript for this line is `reveal`, shown when audio starts."""
        prep = asyncio.create_task(self._prepare_audio(seat, text))
        self._enqueue(("audio", self.speech_epoch, seat, text, seq, prep, reveal))

    def enqueue_event(self, event_type: str, data: Dict[str, Any], seq: int, pace: float = 0) -> None:
        """Shows this after every line already queued, so the transcript stays in spoken order."""
        self._enqueue(("event", self.speech_epoch, event_type, data, seq, pace))

    async def wait_spoken(self) -> None:
        await self._speech_drained.wait()

    async def wait_speech_idle(self) -> None:
        """A human prompt waits until the lines already queued have been spoken."""
        await self.wait_spoken()

    async def _prepare_audio(self, seat: int, text: str) -> Optional[str]:
        if not (self.voice and self.speed <= 2):
            return None
        return await get_tts_audio(with_player_names(text), seat)

    async def _play_speech(self) -> None:
        while True:
            item = await self._speech.get()
            try:
                kind, epoch = item[0], item[1]
                if epoch != self.speech_epoch:
                    if kind == "audio" and not item[5].done():
                        item[5].cancel()
                    continue
                if kind == "event":
                    _, _, event_type, data, seq, pace = item
                    await self.send(event_type, data, seq)
                    if pace:
                        await self.sleep(pace)
                    continue
                _, _, seat, text, seq, prep, reveal = item
                try:
                    wav_b64 = await prep
                except asyncio.CancelledError:
                    raise
                except Exception:
                    wav_b64 = None
                if epoch != self.speech_epoch:
                    continue
                try:
                    await self._emit_audio(seat, text, seq, wav_b64, reveal)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[speech] seat {seat} {type(exc).__name__}: {exc}")
            finally:
                self._speech_pending -= 1
                if self._speech_pending <= 0:
                    self._speech_pending = 0
                    self._speech_drained.set()

    async def speak(self, seat: int, text: str, seq: int) -> None:
        """Voices one line and waits until it finishes. Replay stays in lockstep this way."""
        await self._emit_audio(seat, text, seq, await self._prepare_audio(seat, text))

    async def _emit_audio(
        self,
        seat: int,
        text: str,
        seq: int,
        wav_b64: Optional[str],
        reveal: Optional[tuple] = None,
    ) -> None:
        """Plays one line. `reveal` is the transcript entry, sent as the audio starts."""

        async def begin() -> None:
            if reveal is not None:
                event_type, payload = reveal
                await self.send(event_type, payload, seq)
            await self.send("speaking", {"seat": seat, "active": True}, seq)

        wav = base64.b64decode(wav_b64) if wav_b64 else b""
        epoch = self.speech_epoch
        if wav and self.face_clips:
            if await self._play_stream(seat, wav, seq, begin):
                return
            if epoch != self.speech_epoch or self.closed:
                return
        await begin()
        if wav and avatar_manager.has_seat(seat):
            await avatar_manager.speak(seat, wav)
        elif wav_b64:
            await self.send("audio", {"speaker": seat, "audio": wav_b64, "format": "wav"}, seq)
            await self.sleep(wav_seconds(wav) + 0.3)
        else:
            if seat == self.human_seat:
                await self.sleep(0.8)
            else:
                await self.sleep(min(3.0, 1.0 + 0.05 * len(text.split())))
        await self.send("speaking", {"seat": seat, "active": False}, seq)

    def ack_clip(self, kind: str, seq: object) -> None:
        """The browser reports when this clip actually starts and when it finishes."""
        wait = self._clip_wait
        if wait is None or seq != wait[0]:
            return
        fut = wait[1] if kind == "clip_started" else wait[2]
        if not fut.done():
            fut.set_result(True)

    def _release_clip_wait(self) -> None:
        wait = self._clip_wait
        if wait is None:
            return
        for fut in (wait[1], wait[2]):
            if not fut.done():
                fut.set_result(False)

    async def _play_stream(self, seat: int, wav: bytes, seq: int, begin) -> bool:
        """Shows this line when its clip starts playing, and holds the next line until it ends."""
        epoch = self.speech_epoch
        if self.closed:
            return False
        loop = asyncio.get_running_loop()
        started = loop.create_future()
        finished = loop.create_future()
        self._clip_wait = (seq, started, finished)
        incoming: asyncio.Queue = asyncio.Queue()
        end_mark = object()

        async def pull() -> None:
            try:
                async for chunk in iter_clip_chunks(seat, wav):
                    await incoming.put(chunk)
            finally:
                await incoming.put(end_mark)

        producer = asyncio.create_task(pull())
        getter: Optional[asyncio.Task] = None
        sent = False
        revealed = False
        played = False
        try:
            await self.send("clip_open", {"seat": seat}, seq)
            getter = asyncio.create_task(incoming.get())
            while True:
                if epoch != self.speech_epoch or self.closed:
                    return False
                waiting = {getter}
                if not revealed:
                    waiting.add(started)
                    waiting.add(finished)
                done, _pending = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if not revealed and started.done() and not started.result():
                    return False
                if not revealed and finished.done() and not started.done():
                    return False
                if not revealed and started.done() and started.result():
                    revealed = True
                    await begin()
                if getter not in done:
                    continue
                item = getter.result()
                if item is end_mark:
                    break
                sent = True
                await self.send(
                    "clip_chunk",
                    {"seat": seat, "b64": base64.b64encode(item).decode("ascii")},
                    seq,
                )
                getter = asyncio.create_task(incoming.get())
            if epoch == self.speech_epoch and not self.closed:
                await self.send("clip_end", {"seat": seat}, seq)
            if not sent or epoch != self.speech_epoch or self.closed:
                return False
            if not revealed:
                if finished.done() and not started.done():
                    return False
                try:
                    ok = await asyncio.wait_for(started, 20)
                except asyncio.TimeoutError:
                    ok = False
                if epoch != self.speech_epoch or self.closed or not ok:
                    return False
                await begin()
            if not finished.done():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(finished, wav_seconds(wav) + 3)
            if epoch == self.speech_epoch and not self.closed:
                await self.send("speaking", {"seat": seat, "active": False}, seq)
            played = True
            return True
        finally:
            producer.cancel()
            if getter is not None and not getter.done():
                getter.cancel()
            if not played and not self.closed:
                await self.send("clip_cancel", {"seat": seat}, seq)
            if self._clip_wait is not None and self._clip_wait[0] == seq:
                self._clip_wait = None


# One model call at a time. A new LLM game would otherwise fire a seat's discussion
# and every servant's memory update together.
_LLM_QUEUE = asyncio.Semaphore(1)


class QueuedLLMAgent(LLMAgent):
    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with _LLM_QUEUE:
            return await super()._post(payload)


def roles_payload(roles: Dict[int, Role], sides: Dict[int, bool]) -> Dict[str, Any]:
    return {"roles": {str(s): {"role": ROLE_NAMES[r], "side": "Good" if sides[s] else "Evil"} for s, r in roles.items()}}


@app.websocket("/ws/game")
async def game_ws(ws: WebSocket):
    await ws.accept()
    session = GameSession(ws)
    await session.send("game_list", {"games": list_game_files()})
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "list_games":
                await session.send("game_list", {"games": list_game_files()})
            elif kind == "start_playback":
                session.start(run_playback(session, msg.get("events") or []))
            elif kind == "start_game":
                if msg.get("mode") == "replay":
                    session.start(run_replay(session, msg.get("replay_file") or ""))
                else:
                    human_seat = msg.get("human_seat")
                    players = "llm" if msg.get("players") == "llm" else "heuristic"
                    session.start(run_live_game(session, human_seat=human_seat, players=players))
            elif kind in ("clip_started", "clip_finished"):
                session.ack_clip(kind, msg.get("seq"))
            elif kind == "human_action_response":
                session.resolve_human_action(msg.get("action", ""), msg.get("data", {}))
            elif kind == "stop":
                session.drop_pending_speech()
                if session.task is not None:
                    session.task.cancel()
            elif kind == "settings":
                if "voice" in msg:
                    session.voice = bool(msg["voice"])
            elif kind == "control":
                action = msg.get("action")
                if action == "pause":
                    session.paused = True
                elif action == "resume":
                    session.paused = False
                    session.step()
                elif action == "step":
                    session.step()
                elif action == "speed":
                    session.speed = float(msg.get("value", 1.0))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        session.closed = True
        session.drop_pending_speech()
        session._speech_worker.cancel()
        if session.task is not None:
            session.task.cancel()


# ── Replay ───────────────────────────────────────────────────────────

async def run_replay(session: GameSession, replay_file: str) -> None:
    session.face_clips = False
    path = os.path.normpath(os.path.join(ROOT, replay_file))
    if not path.startswith(ROOT) or not os.path.exists(path):
        await session.send("error", {"message": f"Replay not found: {replay_file}"})
        return
    events = load_events(path)
    adapter = ArenaGraphAdapter(talk_edges_for(path))
    pending_votes: Dict[str, bool] = {}

    for raw in events:
        kind, seq, d = raw["type"], raw["seq"], raw.get("data", {})
        if kind in ("llm_call", "beliefs", "quest_votes", "team_votes"):
            if kind == "team_votes":
                pending_votes = {str(i): bool(v) for i, v in enumerate(d.get("votes", []))}
            adapter.feed(raw)
            continue

        await session.gate()
        stepped = adapter.feed(raw)

        if kind == "game_start":
            state = adapter.state
            await session.send("game_start", {"mode": "replay", "file": replay_file, "leader": d.get("quest_leader", 0)}, seq)
            await session.send("role_reveal", roles_payload(state.roles, adapter.engine.player_sides), seq)
            await session.send("memory_update", {"players": serialize_memories(adapter.memories)}, seq)
            continue

        if kind == "discussion":
            data = {"turn": d["quest"], "round_num": d["round"], "speaker": d["seat"], "text": with_player_names(d["text"])}
            await session.send("utterance", data, seq)
        elif kind == "evil_chat":
            await session.send("evil_chat", {"turn": d.get("quest", 0), "speaker": d["seat"], "text": with_player_names(d["text"])}, seq)
        elif kind == "team_proposed":
            await session.send("team_proposed", {"turn": d["quest"], "round_num": d["round"],
                                                 "leader": d["leader"], "team": d["team"]}, seq)
        elif kind == "team_result":
            await session.send("team_vote_result", {"turn": d["quest"], "round_num": d["round"], "team": d["team"],
                                                    "votes": pending_votes, "approved": d["approved"]}, seq)
            pending_votes = {}
        elif kind == "quest_result":
            await session.send("quest_result", {"turn": d["quest"], "team": d["team"],
                                                "success": d["succeeded"], "num_fails": d["num_fails"]}, seq)
        elif kind == "assassination":
            await session.send("assassination", {"assassin": d.get("assassin"), "target": d.get("target"),
                                                 "hit_merlin": d.get("hit_merlin")}, seq)
        elif kind == "game_end":
            winner = "Good" if str(d.get("winner", "")).lower() == "good" else "Evil"
            await session.send("game_over", {"winner": winner, "reason": d.get("reason", "")}, seq)

        if stepped:
            await session.send("memory_update", {"players": serialize_memories(adapter.memories)}, seq)

        if kind in ("discussion", "evil_chat"):
            await session.speak(d["seat"], with_player_names(d["text"]), seq)
        elif kind in ("team_proposed", "team_result", "quest_result"):
            await session.sleep(1.6)

    await session.send("replay_end", {"violations": adapter.state.violations if adapter.state else []})


# ── Saved live game ──────────────────────────────────────────────────

async def run_playback(session: GameSession, events: List[Any]) -> None:
    """Replays a file exported from a live game: same messages, same graphs, voiced again."""
    session.face_clips = False
    if not isinstance(events, list) or not events:
        await session.send("error", {"message": "That file has no game events."})
        return
    for raw in events[:4000]:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type") or "")
        data = dict(raw.get("data") or {})
        seq = int(raw.get("seq") or 0)
        if kind in ("audio", "avatar_clip", "clip_open", "clip_chunk", "clip_end", "speaking", "game_list", "human_prompt", "human_prompt_clear"):
            continue
        await session.gate()
        if kind in ("utterance", "evil_chat"):
            data["text"] = with_player_names(str(data.get("text") or ""))
            await session.send(kind, data, seq)
            speaker = data.get("speaker")
            if speaker is not None:
                await session.speak(int(speaker), data["text"], seq)
        else:
            await session.send(kind, data, seq)
            if kind in ("team_proposed", "team_vote_result", "quest_result"):
                await session.sleep(1.2)
    await session.send("replay_end", {"violations": []})


# ── Live ─────────────────────────────────────────────────────────────

async def run_live_game(session: GameSession, human_seat: Optional[int] = None, players: str = "heuristic") -> None:
    session.human_seat = human_seat
    session.face_clips = True
    if players == "llm" and not (os.getenv("OPENAI_API_KEY") or os.getenv("BOSON_API_KEY")):
        await session.send("error", {"message": "No OPENAI_API_KEY is set, so an LLM game cannot start."})
        return
    agents: Dict[int, Any] = {}
    for i in range(5):
        if human_seat is not None and i == human_seat:
            session.human_agent = HumanWebAgent(i, session=session, role=Role.LOYAL_SERVANT)
            agents[i] = session.human_agent
        elif players == "llm":
            agents[i] = QueuedLLMAgent(i, Role.LOYAL_SERVANT, condition="belief_graph", seed=None)
        else:
            agents[i] = HeuristicAgent(i, Role.LOYAL_SERVANT, condition="rule_graph", seed=None, merlin_graph=True)

    runner: Optional[AvalonGameRunner] = None
    started = False

    async def announce_start() -> None:
        nonlocal started
        if started:
            return
        started = True
        engine = runner.engine
        await session.send("game_start", {"mode": "live", "leader": engine.leader, "human_seat": human_seat})
        await session.send("role_reveal", roles_payload(engine.player_roles, engine.player_sides))

    def memories():
        return {p: a.memory for p, a in agents.items() if getattr(a, "memory", None) is not None}

    async def on_event(event_type: str, data: Dict[str, Any], recipients: List[int]) -> None:
        await announce_start()
        await session.gate()
        seq = int(data.pop("seq", 0))
        if event_type in ("utterance", "evil_chat"):
            speaker = data.get("speaker")
            # Other players already have the text and can take their turn.
            # The spectator sees the line only when this seat's audio starts.
            data["text"] = with_player_names(str(data.get("text") or ""))
            if speaker != human_seat:
                session.enqueue_speech(speaker, data["text"], seq, (event_type, data))
            else:
                # They already said it, by typing or the mic. Do not voice the line back at them.
                session.enqueue_event(event_type, data, seq)
        elif event_type in ("team_proposed", "team_vote_result", "quest_result"):
            data["votes"] = {str(p): v for p, v in data.get("votes", {}).items()}
            session.enqueue_event(event_type, data, seq, pace=1.6)
            session.enqueue_event("memory_update", {"players": serialize_memories(memories())}, seq)
        elif event_type == "assassination":
            session.enqueue_event("assassination", data, seq, pace=2.0)

    runner = AvalonGameRunner(agents=agents, tts_client=None, verbose=False, on_event=on_event)
    metrics = await runner.run_game()
    session.enqueue_event("memory_update", {"players": serialize_memories(memories())}, runner.seq + 1)
    session.enqueue_event("game_over", {
        "winner": "Good" if metrics.good_victory else "Evil",
        "reason": metrics.win_reason,
    }, runner.seq + 1)
    await session.wait_spoken()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.ws_server:app", host="0.0.0.0", port=8000)
