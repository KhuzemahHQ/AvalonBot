import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type {
  BeliefSnapshot,
  GameEvent,
  GamePhase,
  GameState,
  Player,
  PlayerMemoryView,
  PlayerRole,
  PlayerSide,
  QuestResult,
  TeamProposal,
} from "../types";
import { SEAT_NAMES } from "../types";
import { createClipHub, decodeB64 } from "./clipStream";

type Data = Record<string, unknown>;

const INITIAL_PLAYERS: Player[] = SEAT_NAMES.map((name, seat) => ({
  seat,
  name,
  isSpeaking: false,
  audioLevel: 0,
}));

const INITIAL_STATE: GameState = {
  mode: null,
  phase: "waiting",
  turn: 0,
  roundNum: 0,
  leader: 0,
  players: INITIAL_PLAYERS,
  quests: [null, null, null, null, null],
  currentProposal: null,
  lastVote: null,
  events: [],
  memory: {},
  beliefHistory: {},
  rolesKnown: false,
  violations: [],
  error: null,
  finished: false,
  humanSeat: null,
  humanRole: null,
  humanPrompt: null,
};

type Action =
  | { type: "RESET" }
  | { type: "MESSAGE"; msg: { type: string; seq: number; data: Data } };

const ROLE_KEYS: Record<string, PlayerRole> = {
  merlin: "merlin",
  "loyal servant": "loyal_servant",
  servant: "loyal_servant",
  assassin: "assassin",
  minion: "minion",
};

function votesFrom(raw: unknown): Record<number, boolean> {
  const votes: Record<number, boolean> = {};
  for (const [seat, v] of Object.entries((raw as Record<string, boolean>) ?? {})) votes[Number(seat)] = Boolean(v);
  return votes;
}

function reduceMessage(state: GameState, type: string, seq: number, d: Data): GameState {
  switch (type) {
    case "game_start":
      return {
        ...INITIAL_STATE,
        players: INITIAL_PLAYERS.map((p) => ({ ...p })),
        mode: (d.mode as "live" | "replay") ?? null,
        phase: "team_selection",
        leader: (d.leader as number) ?? 0,
        humanSeat: (d.human_seat as number | null) ?? null,
      };

    case "role_reveal": {
      const roles = (d.roles as Record<string, { role: string; side: string }>) ?? {};
      return {
        ...state,
        rolesKnown: true,
        players: state.players.map((p) => {
          const info = roles[String(p.seat)];
          if (!info) return p;
          return {
            ...p,
            role: ROLE_KEYS[info.role.toLowerCase()],
            side: info.side.toLowerCase() as PlayerSide,
          };
        }),
      };
    }

    case "utterance":
    case "evil_chat": {
      const event = { type, speaker: d.speaker as number, text: d.text as string, seq } as GameEvent;
      return {
        ...state,
        events: [...state.events, event],
        players: state.players.map((p) =>
          p.seat === (d.speaker as number) ? { ...p, lastUtterance: d.text as string } : p
        ),
      };
    }

    case "speaking":
      return {
        ...state,
        players: state.players.map((p) =>
          p.seat === d.seat ? { ...p, isSpeaking: Boolean(d.active) } : d.active ? { ...p, isSpeaking: false } : p
        ),
      };

    case "team_proposed": {
      const proposal: TeamProposal = {
        quest: (d.turn as number) + 1,
        round: d.round_num as number,
        leader: d.leader as number,
        team: d.team as number[],
      };
      return {
        ...state,
        phase: "team_voting",
        turn: d.turn as number,
        roundNum: proposal.round,
        leader: proposal.leader,
        currentProposal: proposal,
        lastVote: null,
        events: [...state.events, { type: "team_proposed", ...proposal, seq }],
      };
    }

    case "team_vote_result": {
      const votes = votesFrom(d.votes);
      const approved = Boolean(d.approved);
      const quest = (d.turn as number) + 1;
      const round = d.round_num as number;
      const voted: TeamProposal = {
        ...(state.currentProposal ?? { quest, round, leader: state.leader, team: (d.team as number[]) ?? [] }),
        votes,
        approved,
      };
      return {
        ...state,
        phase: approved ? "quest_voting" : "team_selection",
        currentProposal: voted,
        lastVote: voted,
        leader: approved ? state.leader : (state.leader + 1) % 5,
        roundNum: approved ? round : round + 1,
        events: [...state.events, { type: "team_vote_result", votes, approved, quest, round, seq }],
      };
    }

    case "quest_result": {
      const idx = d.turn as number;
      const result: QuestResult = {
        quest: idx + 1,
        team: (d.team as number[]) ?? [],
        success: Boolean(d.success),
        numFails: (d.num_fails as number) ?? 0,
        seq,
      };
      const quests = [...state.quests];
      quests[idx] = result;
      const passed = quests.filter((q) => q?.success).length;
      const failed = quests.filter((q) => q && !q.success).length;
      const phase: GamePhase = failed >= 3 ? "game_over" : passed >= 3 ? "assassination" : "team_selection";
      return {
        ...state,
        phase,
        quests,
        turn: Math.min(idx + 1, 4),
        roundNum: 0,
        leader: (state.leader + 1) % 5,
        currentProposal: null,
        events: [...state.events, { type: "quest_result", ...result }],
      };
    }

    case "assassination":
      return {
        ...state,
        events: [
          ...state.events,
          {
            type: "assassination",
            assassin: d.assassin as number,
            target: d.target as number,
            hitMerlin: Boolean(d.hit_merlin),
            seq,
          },
        ],
      };

    case "game_over":
      return {
        ...state,
        phase: "game_over",
        finished: true,
        players: state.players.map((p) => ({ ...p, isSpeaking: false })),
        events: [
          ...state.events,
          {
            type: "game_over",
            winner: String(d.winner).toLowerCase() === "good" ? "good" : "evil",
            reason: (d.reason as string) ?? "",
            seq,
          },
        ],
      };

    case "memory_update": {
      const players = (d.players as Record<string, PlayerMemoryView>) ?? {};
      const memory: Record<number, PlayerMemoryView> = {};
      const beliefHistory = { ...state.beliefHistory };
      for (const [seat, view] of Object.entries(players)) {
        const s = Number(seat);
        memory[s] = view;
        if (!view.beliefName) continue;
        const values: Record<number, number> = {};
        for (const [p, v] of Object.entries(view.beliefs)) values[Number(p)] = v;
        const history = beliefHistory[s] ?? [];
        const last = history[history.length - 1];
        const changed = !last || Object.keys(values).some((p) => last.values[Number(p)] !== values[Number(p)]);
        if (changed || !last) beliefHistory[s] = [...history, { seq, values } as BeliefSnapshot];
      }
      return { ...state, memory, beliefHistory };
    }

    case "human_role": {
      return {
        ...state,
        humanSeat: (d.seat as number) ?? state.humanSeat,
        humanRole: {
          seat: (d.seat as number) ?? 0,
          role: (d.role as string) ?? "",
          side: String(d.side ?? "good").toLowerCase() as "good" | "evil",
          seenPlayers: (d.seen_players as number[]) ?? [],
          evilPlayers: (d.evil_players as number[]) ?? [],
        },
      };
    }

    case "human_prompt": {
      return {
        ...state,
        humanPrompt: {
          action: (d.action as any) ?? "discussion",
          seat: (d.seat as number) ?? 0,
          turn: d.turn as number | undefined,
          roundNum: d.round_num as number | undefined,
          leader: d.leader as number | undefined,
          team: d.team as number[] | undefined,
          teamSize: d.team_size as number | undefined,
          canFail: d.can_fail as boolean | undefined,
          side: d.side as "good" | "evil" | undefined,
          candidates: d.candidates as number[] | undefined,
        },
      };
    }

    case "human_prompt_clear":
      return { ...state, humanPrompt: null };

    case "replay_end":
      return { ...state, finished: true, violations: (d.violations as string[]) ?? [] };

    case "error":
      return { ...state, error: (d.message as string) ?? "Unknown error" };

    default:
      return state;
  }
}

function gameReducer(state: GameState, action: Action): GameState {
  if (action.type === "RESET") return { ...INITIAL_STATE, players: INITIAL_PLAYERS.map((p) => ({ ...p })) };
  const { type, seq, data } = action.msg;
  return reduceMessage(state, type, seq, data);
}

interface UseGameSocketReturn {
  gameState: GameState;
  isConnected: boolean;
  startGame: (mode: "live" | "replay", replayFile?: string, humanSeat?: number | null, players?: "llm" | "heuristic") => void;
  sendHumanAction: (action: string, data: Record<string, unknown>) => void;
  stopGame: () => void;
  pause: () => void;
  resume: () => void;
  step: () => void;
  setSpeed: (speed: number) => void;
  setVoice: (on: boolean) => void;
  availableGames: string[];
  seatClips: Record<number, string>;
  seatClipSeq: Record<number, number>;
  reportClip: (seq: number, phase: "started" | "finished") => void;
  exportGame: () => void;
  playUpload: (events: { type: string; seq?: number; data?: Data }[]) => void;
}

export function useGameSocket(
  wsUrl = "/ws/game",
  onAudio?: (seat: number, base64Wav: string) => void
): UseGameSocketReturn {
  const [state, dispatch] = useReducer(gameReducer, INITIAL_STATE);
  const [isConnected, setIsConnected] = useState(false);
  const [availableGames, setAvailableGames] = useState<string[]>([]);
  const [seatClips, setSeatClips] = useState<Record<number, string>>({});
  const [seatClipSeq, setSeatClipSeq] = useState<Record<number, number>>({});
  const seatClipsRef = useRef<Record<number, string>>({});
  seatClipsRef.current = seatClips;
  const recordingRef = useRef<{ type: string; seq: number; data: Data }[]>([]);
  const clipHubRef = useRef<ReturnType<typeof createClipHub> | null>(null);
  if (!clipHubRef.current) {
    clipHubRef.current = createClipHub((seat, url, seq) => {
      setSeatClips((prev) => ({ ...prev, [seat]: url }));
      setSeatClipSeq((prev) => ({ ...prev, [seat]: seq }));
    });
  }
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay = useRef(1000);
  const onAudioRef = useRef(onAudio);
  onAudioRef.current = onAudio;
  const unmounted = useRef(false);

  const send = useCallback((msg: Data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify(msg));
  }, []);

  const connect = useCallback(() => {
    if (unmounted.current || wsRef.current) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = wsUrl.startsWith("/") ? `${protocol}//${window.location.host}${wsUrl}` : wsUrl;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (wsRef.current !== ws) return;
      setIsConnected(true);
      reconnectDelay.current = 1000;
    };

    ws.onmessage = (event) => {
      if (wsRef.current !== ws) return;
      let msg: { type: string; seq?: number; data?: Data };
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      const data = msg.data ?? {};
      if (msg.type === "game_list") {
        setAvailableGames((data.games as string[]) ?? []);
        return;
      }
      if (msg.type === "audio") {
        onAudioRef.current?.(data.speaker as number, data.audio as string);
        return;
      }
      if (
        msg.type !== "audio" &&
        msg.type !== "avatar_clip" &&
        msg.type !== "clip_open" &&
        msg.type !== "clip_chunk" &&
        msg.type !== "clip_end" &&
        msg.type !== "clip_cancel" &&
        msg.type !== "speaking" &&
        msg.type !== "game_list" &&
        msg.type !== "human_prompt" &&
        msg.type !== "human_prompt_clear"
      ) {
        recordingRef.current.push({ type: msg.type, seq: msg.seq ?? 0, data });
      }
      if (msg.type === "clip_open") {
        clipHubRef.current?.open(Number(data.seat), msg.seq ?? 0);
        return;
      }
      if (msg.type === "clip_chunk") {
        const b64 = String(data.b64 ?? "");
        if (b64) clipHubRef.current?.chunk(Number(data.seat), msg.seq ?? 0, decodeB64(b64));
        return;
      }
      if (msg.type === "clip_end") {
        clipHubRef.current?.end(Number(data.seat), msg.seq ?? 0);
        return;
      }
      if (msg.type === "clip_cancel") {
        const seat = Number(data.seat);
        clipHubRef.current?.drop(seat, msg.seq ?? 0);
        setSeatClips((prev) => {
          const next = { ...prev };
          delete next[seat];
          return next;
        });
        setSeatClipSeq((prev) => {
          const next = { ...prev };
          delete next[seat];
          return next;
        });
        return;
      }
      if (msg.type === "avatar_clip") {
        const seat = Number(data.seat);
        const video = String(data.video ?? "");
        if (!video) return;
        const binary = atob(video);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const url = URL.createObjectURL(new Blob([bytes], { type: "video/mp4" }));
        const previous = seatClipsRef.current[seat];
        if (previous) URL.revokeObjectURL(previous);
        setSeatClips((prev) => ({ ...prev, [seat]: url }));
        return;
      }
      dispatch({ type: "MESSAGE", msg: { type: msg.type, seq: msg.seq ?? 0, data } });
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      setIsConnected(false);
      wsRef.current = null;
      if (unmounted.current) return;
      reconnectTimer.current = setTimeout(() => {
        reconnectDelay.current = Math.min(reconnectDelay.current * 1.5, 16000);
        connect();
      }, reconnectDelay.current);
    };

    ws.onerror = () => ws.close();
  }, [wsUrl]);

  useEffect(() => {
    unmounted.current = false;
    connect();
    return () => {
      unmounted.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  const startGame = useCallback(
    (mode: "live" | "replay", replayFile?: string, humanSeat?: number | null, players?: "llm" | "heuristic") => {
      dispatch({ type: "RESET" });
      recordingRef.current = [];
      clipHubRef.current?.reset();
      setSeatClips({});
      setSeatClipSeq({});
      send({ type: "start_game", mode, replay_file: replayFile, human_seat: humanSeat, players });
    },
    [send]
  );

  const sendHumanAction = useCallback(
    (action: string, data: Record<string, unknown>) => {
      send({ type: "human_action_response", action, data });
      dispatch({ type: "MESSAGE", msg: { type: "human_prompt_clear", seq: 0, data: {} } });
    },
    [send]
  );

  return {
    gameState: state,
    isConnected,
    startGame,
    sendHumanAction,
    stopGame: useCallback(() => send({ type: "stop" }), [send]),
    pause: useCallback(() => send({ type: "control", action: "pause" }), [send]),
    resume: useCallback(() => send({ type: "control", action: "resume" }), [send]),
    step: useCallback(() => send({ type: "control", action: "step" }), [send]),
    setSpeed: useCallback((speed: number) => send({ type: "control", action: "speed", value: speed }), [send]),
    setVoice: useCallback((on: boolean) => send({ type: "settings", voice: on }), [send]),
    availableGames,
    seatClips,
    seatClipSeq,
    reportClip: useCallback((seq: number, phase: "started" | "finished") => {
      send({ type: phase === "started" ? "clip_started" : "clip_finished", seq });
    }, [send]),
    exportGame: useCallback(() => {
      const payload = JSON.stringify({ version: 1, events: recordingRef.current });
      const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = "avalon-game.json";
      link.click();
      URL.revokeObjectURL(url);
    }, []),
    playUpload: useCallback((events: { type: string; seq?: number; data?: Data }[]) => {
      dispatch({ type: "RESET" });
      recordingRef.current = [];
      clipHubRef.current?.reset();
      setSeatClips({});
      setSeatClipSeq({});
      send({ type: "start_playback", events });
    }, [send]),
  };
}
