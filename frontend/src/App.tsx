import { useCallback, useEffect, useMemo, useState } from "react";
import GameTable from "./components/GameTable";
import QuestTracker from "./components/QuestTracker";
import VoteOverlay from "./components/VoteOverlay";
import ChatLog from "./components/ChatLog";
import PhaseIndicator from "./components/PhaseIndicator";
import GameControls from "./components/GameControls";
import KnowledgeGraphView from "./components/KnowledgeGraphView";
import { BeliefPanel } from "./components/BeliefPanel";
import HumanActionModal from "./components/HumanActionModal";
import { useGameSocket } from "./hooks/useGameSocket";
import { useAudio } from "./hooks/useAudio";
import { seatName } from "./types";

type View = "table" | "graphs";

function TopButton({ children, onClick, active = false, disabled = false, title }: {
  children: React.ReactNode;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`rounded-full border px-3.5 py-1.5 font-body text-xs transition-all duration-300 ease-fluid active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50 ${
        active
          ? "border-medieval-gold/50 bg-medieval-gold/10 text-medieval-gold"
          : "border-medieval-border bg-medieval-darker text-medieval-parchment-dim hover:border-medieval-gold/30 hover:text-medieval-gold"
      }`}
    >
      {children}
    </button>
  );
}

export default function App() {
  const [view, setView] = useState<View>("table");
  const [showRoles, setShowRoles] = useState(false);
  const [speed, setSpeedLocal] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [voiceOn, setVoiceOn] = useState(true);

  const audio = useAudio();
  const { gameState, isConnected, startGame, sendHumanAction, pause, resume, step, setSpeed, setVoice, availableGames, seatClips, seatClipSeq, reportClip, exportGame, playUpload } =
    useGameSocket("/ws/game", audio.playAudio);
  const [faceStamp, setFaceStamp] = useState("");
  useEffect(() => {
    let stop = false;
    const tick = () => {
      fetch("/api/faces/stamp")
        .then((r) => r.json())
        .then((d: { stamp?: string }) => {
          if (!stop && d.stamp) setFaceStamp(d.stamp);
        })
        .catch(() => undefined);
    };
    tick();
    const id = setInterval(tick, 20000);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, []);

  const phaseDetails = useMemo(() => {
    const { phase, leader, turn, roundNum, currentProposal, mode } = gameState;
    switch (phase) {
      case "waiting":
        return isConnected ? "Start a live game or watch a recorded one below" : "Connecting to the game server…";
      case "team_selection":
        return `Quest ${turn + 1} · Proposal ${roundNum + 1} · ${seatName(leader)} leads${mode === "replay" ? " · replay" : ""}`;
      case "team_voting":
        return currentProposal
          ? `Quest ${currentProposal.quest} · voting on ${currentProposal.team.map(seatName).join(", ")}`
          : "Voting";
      case "quest_voting":
        return `Quest ${turn + 1} · the team plays its cards`;
      case "assassination":
        return "Good won three quests · the Assassin hunts for Merlin";
      case "game_over": {
        const end = gameState.events.find((e) => e.type === "game_over");
        return end?.type === "game_over" ? end.reason : "Game complete";
      }
    }
  }, [gameState, isConnected]);

  const winner = useMemo(() => {
    const end = gameState.events.find((e) => e.type === "game_over");
    return end?.type === "game_over" ? end.winner : null;
  }, [gameState.events]);

  const currentSpeaker = gameState.players.find((p) => p.isSpeaking)?.seat ?? null;
  const playing = isPlaying && !gameState.finished;

  const handlePlayPause = useCallback(() => {
    if (playing) {
      pause();
      setIsPlaying(false);
    } else {
      resume();
      setIsPlaying(true);
    }
  }, [playing, pause, resume]);

  const begin = useCallback(
    (mode: "live" | "replay", file?: string, humanSeat?: number | null, players?: "llm" | "heuristic") => {
      audio.unlock();
      startGame(mode, file, humanSeat, players);
      setSpeed(speed);
      setVoice(voiceOn);
      setIsPlaying(true);
    },
    [audio, startGame, setSpeed, setVoice, speed, voiceOn]
  );

  const [autoplayed, setAutoplayed] = useState(false);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const auto = params.get("autoplay");
    if (autoplayed || !isConnected || !auto) return;
    if (auto === "replay" && availableGames.length === 0) return;
    setAutoplayed(true);
    const v = params.get("view");
    if (v === "graphs" || v === "table") setView(v);
    if (params.get("roles") === "1") setShowRoles(true);
    const s = Number(params.get("speed"));
    if (s > 0) setSpeedLocal(s);
    startGame(auto === "live" ? "live" : "replay", params.get("file") ?? availableGames[0]);
    if (s > 0) setSpeed(s);
    setVoice(params.get("voice") !== "0" && voiceOn);
    setIsPlaying(true);
  }, [autoplayed, isConnected, availableGames, startGame, setSpeed, setVoice, voiceOn]);

  const toggleMute = useCallback(() => {
    audio.setMuted(!audio.isMuted);
  }, [audio]);

  const hasGame = gameState.mode !== null;
  const status = hasGame
    ? `${gameState.mode === "replay" ? "Replay" : "Live"} · ${gameState.events.length} events${gameState.finished ? " · finished" : ""}`
    : "";

  return (
    <div className="flex min-h-[100dvh] flex-col" style={{ background: "radial-gradient(ellipse at 50% 20%, rgba(40, 32, 64, 0.45) 0%, #0a0a0f 60%)" }}>
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-medieval-border/50 px-6 py-3">
        <div className="flex items-center gap-3">
          <img src="/crest.webp" alt="" className="h-9 w-9 object-contain" />
          <div>
            <div className="font-display text-base font-semibold tracking-[0.18em] text-medieval-gold">AVALON</div>
            <div className="font-body text-[10px] uppercase tracking-[0.2em] text-medieval-silver/70">AI spectator table</div>
          </div>
        </div>

        <div className="flex rounded-full border border-medieval-border bg-medieval-darker p-1">
          {(["table", "graphs"] as View[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`rounded-full px-4 py-1.5 font-body text-xs transition-all duration-300 ${
                view === v ? "bg-medieval-gold/15 text-medieval-gold" : "text-medieval-silver hover:text-medieval-parchment"
              }`}
            >
              {v === "table" ? "Round Table" : "Knowledge graphs"}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="mr-1 flex items-center gap-1.5 font-body text-[11px] text-medieval-silver">
            <span className={`h-2 w-2 rounded-full ${isConnected ? "bg-medieval-good" : "bg-medieval-crimson-bright"}`} />
            {isConnected ? "Server" : "Offline"}
          </span>
          {gameState.humanRole && (
            <div className="flex items-center gap-2 rounded-full border border-medieval-gold/40 bg-medieval-gold/10 px-3.5 py-1 text-xs">
              <span className="font-display font-semibold text-medieval-gold">
                👑 Secret Identity: {gameState.humanRole.role} ({gameState.humanRole.side === "good" ? "Good" : "Evil"})
              </span>
              {gameState.humanRole.seenPlayers?.length > 0 && (
                <span className="font-mono text-[10px] text-medieval-silver/80">
                  Seen Evil: {gameState.humanRole.seenPlayers.map(seatName).join(", ")}
                </span>
              )}
              {gameState.humanRole.evilPlayers?.length > 0 && (
                <span className="font-mono text-[10px] text-medieval-silver/80">
                  Coven: {gameState.humanRole.evilPlayers.filter((s) => s !== gameState.humanSeat).map(seatName).join(", ")}
                </span>
              )}
            </div>
          )}
          <TopButton onClick={() => setShowRoles(!showRoles)} active={showRoles} disabled={!gameState.rolesKnown} title="Spectators only: the players never see this">
            {showRoles ? "Hide roles" : "Reveal roles"}
          </TopButton>
          <TopButton onClick={toggleMute} active={audio.isMuted}>
            {audio.isMuted ? "Unmute" : "Mute"}
          </TopButton>
        </div>
      </header>

      <PhaseIndicator phase={gameState.phase} details={phaseDetails} winner={winner} />

      {(gameState.error || gameState.violations.length > 0) && (
        <div className="mx-auto mt-3 flex w-full max-w-[1500px] flex-col gap-1 px-6 font-body text-xs">
          {gameState.error && <div className="rounded-lg border border-medieval-crimson-bright/40 bg-[#2a0d0d] px-3 py-2 text-[#f08a8a]">{gameState.error}</div>}
          {gameState.violations.length > 0 && (
            <div className="rounded-lg border border-[#e0a83a]/30 bg-[#2a210d] px-3 py-2 text-[#e0c08a]">
              Rule violations in the recorded game: {gameState.violations.join("; ")}
            </div>
          )}
        </div>
      )}

      {/* Interactive Human Action Modal for Proposals, Votes, Secret Quest Cards, & Assassination */}
      {gameState.humanPrompt && (
        <HumanActionModal
          prompt={gameState.humanPrompt}
          onRespond={sendHumanAction}
        />
      )}

      <main
        className={`mx-auto grid min-h-0 w-full max-w-[1500px] flex-1 grid-cols-1 gap-5 px-6 py-5 ${
          view === "table" ? "lg:grid-cols-[300px_minmax(0,1fr)_340px]" : "lg:grid-cols-[minmax(0,1fr)_340px]"
        }`}
      >
        {view === "table" ? (
          <>
            <aside className="order-2 flex flex-col gap-4 lg:order-1">
              <QuestTracker quests={gameState.quests} currentQuest={gameState.turn + 1} />
              <BeliefPanel memory={gameState.memory} history={gameState.beliefHistory} quests={gameState.quests} />
            </aside>
            <section className="order-1 flex items-center justify-center lg:order-2">
              <GameTable
                players={gameState.players}
                leader={gameState.leader}
                proposal={gameState.currentProposal}
                showRoles={showRoles}
                seatClips={seatClips}
                seatClipSeq={seatClipSeq}
                onClipPhase={reportClip}
                faceStamp={faceStamp}
                muted={audio.isMuted}
                levels={audio.levels}
              >
                <VoteOverlay events={gameState.events} />
              </GameTable>
            </section>
          </>
        ) : (
          <section className="flex flex-col gap-4">
            <KnowledgeGraphView memory={gameState.memory} players={gameState.players} showRoles={showRoles} />
          </section>
        )}

        <aside className="order-3 flex max-h-[calc(100dvh-210px)] min-h-[420px] flex-col overflow-hidden rounded-2xl border border-medieval-border bg-medieval-darker">
          <ChatLog events={gameState.events} currentSpeaker={currentSpeaker} />
        </aside>
      </main>

      <GameControls
        isPlaying={playing}
        hasGame={hasGame && !gameState.finished}
        speed={speed}
        voiceOn={voiceOn}
        onPlayPause={handlePlayPause}
        onStep={step}
        onSpeedChange={(s) => {
          setSpeedLocal(s);
          setSpeed(s);
        }}
        onVoiceToggle={() => {
          setVoiceOn(!voiceOn);
          setVoice(!voiceOn);
        }}
        onNewGame={(humanSeat) => begin("live", undefined, humanSeat)}
        onLlmGame={() => begin("live", undefined, null, "llm")}
        onLoadReplay={(file) => begin("replay", file)}
        onExport={exportGame}
        canExport={gameState.finished}
        onPlayFile={(file) => {
          file.text().then((text) => {
            const parsed = JSON.parse(text) as { events?: { type: string; seq?: number; data?: Record<string, unknown> }[] } | { type: string }[];
            const events = Array.isArray(parsed) ? parsed : parsed.events;
            if (Array.isArray(events) && events.length) {
              audio.unlock();
              playUpload(events as { type: string; seq?: number; data?: Record<string, unknown> }[]);
              setIsPlaying(true);
            }
          }).catch(() => undefined);
        }}
        status={status}
        availableReplays={availableGames}
      />
    </div>
  );
}
