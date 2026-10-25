"""
LLM-powered Agent for 5-Player Avalon.

Every agent reads only its own memory (see MEMORY_MODEL.md). For Loyal Servants:
- Condition A (chat_only): full observation log plus free-text notes from a reflection call
- Condition B (belief_graph): graph table plus recall tools; P(evil) from its own extraction call
- Condition C (rule_graph): graph table plus recall tools; P(evil) from the Condition C rule
Merlin and the evil players see their full observation log in every condition.
Compatible with OpenAI, Boson AI, Groq, or any OpenAI-compatible API endpoint.
"""

import os
import json
import math
from typing import Dict, List, Optional, Any
import aiohttp

from .engine import Role, Phase, ROLE_NAMES, ROLE_SIDES
from .agents import Agent
from .tools import RECALL_TOOL_DEFINITIONS, get_tool


SYSTEM_PROMPT_TEMPLATE = """You are Player {player_id} in a 5-player game of The Resistance: Avalon.
Your Secret Role: {role_name} ({side_name})

Role rules:
- Good Team (3 players): Merlin, 2 Loyal Servants. Goal: pass 3 quests and keep Merlin secret.
- Evil Team (2 players): Minion, Assassin. Goal: fail 3 quests OR assassinate Merlin at the end.
- Merlin sees who is Evil ({seen_evil_str}), but MUST NOT reveal this identity directly or the Assassin will shoot you!
- Evil players see each other ({seen_evil_str}).
- Servants do not know anyone's identity and must deduce who is evil through voting and proposals.
- A failed quest proves at least one team member is evil. A passed quest proves nothing, since evil may pass.

Your memory below holds only what you observed and what you yourself thought. Entries are numbered (#).

Personality:
- Be concise, strategic, and direct. Keep your spoken lines to 1-2 natural sentences.
- Never use markdown formatting, bullet points, or emojis in spoken discussion.
"""

EXTRACTION_PROMPT = """Discussion round {label} just ended. Record what you heard into your private memory by calling record_observations.

This round's lines:
{transcript}

Rules:
- edges: only for what players actually said. Cite the # numbers of the speaker's own lines in sources. Quote at most 10 words.
- accused/defended point at a player id. backed_team/opposed_team point at an existing team label like Q1.1. claimed_role names a role.
- notes: your own read on each player, at most 15 words each.
{p_evil_rule}"""

P_EVIL_RULE_B = (
    "- p_evil: your probability that each of players {opponents} is evil. Exactly 2 of them are evil, so the values "
    "sum to 2. Each failed team's members must sum to at least its number of fails. Votes and quest results "
    "outweigh talk."
)
P_EVIL_RULE_C = "- Leave out p_evil. Your beliefs are computed by code from votes, quest results, and these edges."

MOCK_LINE = "I support this mission and want to ensure we pick trusted players."
MAX_RECALL_STEPS = 2


class LLMAgent(Agent):
    """
    LLM-powered Avalon agent for conditions A, B, and C.
    """

    llm_beliefs = True

    def __init__(
        self,
        player_id: int,
        role: Role,
        num_players: int = 5,
        condition: str = "belief_graph",  # "chat_only", "belief_graph", or "rule_graph"
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        merlin_graph: bool = False,
        transport=None,
        percival_condition: str = "chat_only",
    ):
        super().__init__(player_id, role, num_players, condition, seed, merlin_graph=merlin_graph, percival_condition=percival_condition)
        # Unset arguments come from the standard OpenAI variables, so a local .env can route every
        # agent to an OpenAI-compatible endpoint (e.g. Boson Marketplace) without code changes.
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("BOSON_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.transport = transport
        self.temperature = temperature

    @property
    def _offers_recall(self) -> bool:
        return self.memory is not None and ((self.is_servant and self.uses_graph) or self.memory.percival is not None)

    def _build_system_prompt(self) -> str:
        seen_evil = self.private_info.get("seen_evil", [])
        seen_evil_str = f"Player(s) {seen_evil}" if seen_evil else "none"

        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            player_id=self.player_id,
            role_name=ROLE_NAMES[self.role],
            side_name="Good" if self.side else "Evil",
            seen_evil_str=seen_evil_str,
        )
        if self.private_info.get("percival_game"):
            prompt = prompt.replace("Merlin, 2 Loyal Servants", "Merlin, Percival, Loyal Servant").replace("Minion, Assassin", "Morgana, Assassin")
            if self.role == Role.MERLIN:
                prompt += "\nHide Merlin: blend with the other good voters' public accuracy instead of standing out as the most accurate. Avoid rejecting every team containing Morgana; use public evidence to justify your decisions. Do not reveal your role or private knowledge."
            elif self.role == Role.MORGANA:
                prompt += "\nImitate Merlin to mislead Percival: hint at knowledge, reject obviously evil teams early when useful, and still seek quest seats. Do not reveal your evil role."
            elif self.role == Role.PERCIVAL:
                prompt += "\nYour two private candidates are Merlin and Morgana, without labels. Speak confidently, citing public evidence only, to act as a decoy Merlin. Never name Merlin or disclose either candidate, even when certain."
        return prompt

    def _memory_context(self) -> str:
        m = self.memory
        if m is None:
            return ""
        parts = [m.render_private()]
        if m.percival is not None and m.percival.following() is not None:
            parts.append(f"Privately follow Player {m.percival.following()}'s publicly observed proposals and votes when choosing teams and discussing. Never disclose this inference. Team votes still use your all-good probability cutoff.")
        if self._offers_recall:
            parts.append("Your belief graph:\n" + m.render_table())
            quest = m.log[-1].quest if m.log else 1
            recent = m.render_log(m.entries_since_quest(quest - 1))
            parts.append("Recent events (use the recall tools for older ones):\n" + (recent or "none yet"))
        else:
            if m.merlin is not None:
                parts.append(
                    "Your Merlin graph (P(Merlin): how likely each good player is Merlin, "
                    "based on how accurate their votes and accusations have been):\n" + m.render_table()
                )
            parts.append("Everything you have observed:\n" + (m.render_log() or "nothing yet"))
            if m.free_notes:
                parts.append("Your private notes:\n" + m.render_notes())
            parts.append(m.graph.render_my_record())
        return "\n\n".join(parts)

    def _messages(self, user_prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._memory_context() + "\n\n" + user_prompt},
        ]

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.transport is not None:
            return await self.transport.chat(payload)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=45)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    err_msg = await resp.text()
                    raise RuntimeError(f"LLM API Error ({resp.status}): {err_msg}")
                return await resp.json()

    @staticmethod
    def _parse_args(call: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(call["function"].get("arguments") or "{}")
        except (KeyError, TypeError, ValueError):
            return {}

    async def _call_llm(self, user_prompt: str, json_mode: bool = False, max_tokens: int = 150) -> str:
        """
        Executes an asynchronous chat completion request.
        """
        if not self.api_key and self.transport is None:
            # Fallback mock for testing without live API keys
            return MOCK_LINE

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(user_prompt),
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = await self._post(payload)
        return data["choices"][0]["message"]["content"].strip()

    async def _call_llm_tool(self, user_prompt: str, tool_name: str) -> Dict[str, Any]:
        """
        Calls the LLM until it uses tool_name. Servants in B and C may first use the
        recall tools over their own memory, for up to MAX_RECALL_STEPS rounds.
        """
        if not self.api_key and self.transport is None:
            return {}

        messages = self._messages(user_prompt)
        tools = [get_tool(tool_name)]

        if self._offers_recall:
            tools = tools + RECALL_TOOL_DEFINITIONS
            try:
                for _ in range(MAX_RECALL_STEPS):
                    data = await self._post({
                        "model": self.model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "required",
                        "temperature": self.temperature,
                    })
                    msg = data["choices"][0]["message"]
                    calls = msg.get("tool_calls") or []
                    if not calls:
                        break
                    for call in calls:
                        if call["function"]["name"] == tool_name:
                            return self._parse_args(call)
                    messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
                    for call in calls:
                        result = self.memory.run_tool(call["function"]["name"], self._parse_args(call))
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            except RuntimeError:
                pass  # Provider may not support tool_choice "required"; force the tool below.

        data = await self._post({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "temperature": self.temperature,
        })
        choice = data["choices"][0]["message"]
        for call in choice.get("tool_calls") or []:
            if call["function"]["name"] == tool_name:
                return self._parse_args(call)
        return {}

    async def after_discussion(self, turn: int, round_num: int) -> None:
        """
        This player's own read of the discussion round. Servants: reflection in A, extraction
        in B and C. Evil players with a Merlin graph: extraction for talk edges only.
        """
        if self.memory is None or not (self.is_servant or self.role == Role.PERCIVAL or self.reads_talk):
            return
        m = self.memory
        label = f"Q{turn + 1}.{round_num + 1}"
        transcript = m.render_log(m.discussion_entries(turn + 1, round_num + 1)) or "Nobody spoke."

        if not self.reads_talk:
            if not self.api_key and self.transport is None:
                return
            prompt = (
                f"Private reflection after discussion round {label}. Nobody else will see this. "
                f"In at most 40 words, note who you suspect and why, citing # numbers.\n\nThis round:\n{transcript}"
            )
            try:
                m.add_free_note(await self._call_llm(prompt, max_tokens=80))
            except Exception:
                pass
            return

        opponents = [p for p in range(self.num_players) if p != self.player_id]
        llm_sets_p_evil = m.beliefs is not None and m.belief_mode == "llm"
        p_evil_rule = P_EVIL_RULE_B.format(opponents=opponents) if llm_sets_p_evil else P_EVIL_RULE_C
        prompt = EXTRACTION_PROMPT.format(label=label, transcript=transcript, p_evil_rule=p_evil_rule)
        try:
            args = await self._call_llm_tool(prompt, "record_observations")
        except Exception:
            args = {}
        m.add_talk_edges(args.get("edges") or [])
        m.set_notes(args.get("notes") or {})

        if not llm_sets_p_evil:
            return
        # Beyond the coherence rules, reject P(evil) that ignores the failed quests (all 0.5, or a
        # failed team not ranked as the top suspects). Re-ask once, then use the pair-counting rule.
        errors = m.apply_llm_beliefs(args.get("p_evil"), check_evidence=True)
        if not errors:
            m.record_belief_path("llm")
            return
        retry = (
            prompt + "\n\nYour p_evil was rejected: " + "; ".join(errors)
            + ". Call record_observations again with corrected p_evil."
        )
        try:
            retry_args = await self._call_llm_tool(retry, "record_observations")
        except Exception:
            retry_args = {}
        if m.apply_llm_beliefs(retry_args.get("p_evil"), check_evidence=True):
            m.fall_back_to_rule()
            m.record_belief_path("rule_fallback")
        else:
            m.record_belief_path("llm_retry")

    async def final_beliefs(self) -> Optional[Dict[int, float]]:
        m = self.memory
        if m is not None and m.percival is not None:
            return dict(m.percival.p_evil)
        if m is None or m.beliefs is None:
            return None
        if self.uses_graph:
            return dict(m.beliefs.p_evil)

        # Condition A keeps no beliefs during play; it is asked once, at game end.
        if self.api_key or self.transport is not None:
            opponents = [p for p in range(self.num_players) if p != self.player_id]
            prompt = (
                f"The game is over. Call report_beliefs with your probability that each of players {opponents} "
                f"is evil. Exactly 2 of them are evil, so the values sum to 2."
            )
            try:
                args = await self._call_llm_tool(prompt, "report_beliefs")
            except Exception:
                args = {}
            if not m.apply_llm_beliefs(args.get("p_evil")):
                return dict(m.beliefs.p_evil)
        m.fall_back_to_rule()
        return dict(m.beliefs.p_evil)

    async def final_merlin_beliefs(self):
        if self.role != Role.PERCIVAL or self.memory is None:
            return None
        if self.memory.percival is not None:
            return dict(self.memory.percival.p_merlin)
        candidates = self.private_info["merlin_candidates"]
        args = await self._call_llm_tool(
            f"Game over. Privately call report_merlin with probabilities for candidates {candidates}, summing to 1. Nobody else sees this.", "report_merlin")
        try:
            probabilities = {int(p): float(v) for p, v in args["probabilities"].items()}
            if set(probabilities) != set(candidates) or any(not math.isfinite(v) or v < 0 or v > 1 for v in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.01):
                return None
            total = sum(probabilities.values())
            return {p: v / total for p, v in probabilities.items()}
        except (KeyError, TypeError, ValueError):
            return None

    async def evil_discussion(self, turn: int) -> str:
        prompt = (
            "Good has passed 3 quests. Only your evil partner hears this. "
            "In 1-2 sentences, say who you think Merlin is and why."
        )
        try:
            return await self._call_llm(prompt)
        except Exception:
            return ""

    async def team_discussion(self, turn: int, round_num: int, leader: int, team_size: int) -> str:
        prompt = (
            f"Current state: Quest {turn + 1}, Proposal Round {round_num + 1}. Leader is Player {leader}.\n"
            f"Say 1-2 spoken sentences to the table. State who you trust or suspect, or advise on the team selection."
        )
        try:
            return await self._call_llm(prompt)
        except Exception as e:
            return f"As Player {self.player_id}, I want to make sure we keep evil players off Quest {turn + 1}."

    def _valid_team(self, team: List[int], team_size: int) -> bool:
        return len(set(team)) == team_size and all(0 <= p < self.num_players for p in team)

    async def propose_team(self, turn: int, team_size: int) -> List[int]:
        round_num = self._current_round_num()
        prompt = f"You are the Leader for Quest {turn + 1}. Call propose_team with exactly {team_size} distinct players."
        try:
            # 1. Try Tool Calling first
            args = await self._call_llm_tool(prompt, "propose_team")
            if "team" in args:
                team = [int(x) for x in args["team"]]
                if self._valid_team(team, team_size):
                    self._record_reason(turn, round_num, "proposed", str(args.get("reasoning", "")))
                    return sorted(team)

            # 2. Try JSON mode fallback
            resp = await self._call_llm(
                prompt + " Output strictly JSON: {\"team\": [...], \"reasoning\": \"...\"}.", json_mode=True
            )
            data = json.loads(resp)
            team = [int(x) for x in data["team"]]
            if self._valid_team(team, team_size):
                self._record_reason(turn, round_num, "proposed", str(data.get("reasoning", "")))
                return sorted(team)
        except Exception:
            pass

        # 3. Fallback: least suspected players, when this player has an estimate
        team = [self.player_id]
        others = [p for p in range(self.num_players) if p != self.player_id]
        if all(self.suspicion(p) is not None for p in others):
            others.sort(key=lambda p: self.suspicion(p))
        team.extend(others[: team_size - 1])
        self._record_reason(turn, round_num, "proposed", "fallback: picked the players I suspect least")
        return sorted(team)

    async def vote_on_team(self, turn: int, round_num: int, leader: int, team: List[int]) -> bool:
        if round_num >= 4:
            self._record_reason(turn, round_num, "approved", "hammer round auto-approves")
            return True  # Hammer round auto-approves

        if self.memory is not None and self.memory.percival is not None:
            from .percival import TEAM_APPROVAL_THRESHOLD
            probability = self.memory.percival.all_good(team)
            approve = probability > TEAM_APPROVAL_THRESHOLD
            self._record_reason(turn, round_num, "approved" if approve else "rejected", f"P(all good)={probability:.4f}")
            return approve

        prompt = f"Quest {turn + 1}, Round {round_num + 1}. Leader Player {leader} proposed team {team}. Call vote_on_team to approve or reject."
        try:
            # 1. Try Tool Calling first
            args = await self._call_llm_tool(prompt, "vote_on_team")
            if "approve" in args:
                approve = bool(args["approve"])
                self._record_reason(turn, round_num, "approved" if approve else "rejected", str(args.get("reason", "")))
                return approve

            # 2. Try JSON mode fallback
            resp = await self._call_llm(prompt + " Output JSON: {\"approve\": true/false, \"reason\": \"...\"}", json_mode=True)
            data = json.loads(resp)
            approve = bool(data.get("approve", True))
            self._record_reason(turn, round_num, "approved" if approve else "rejected", str(data.get("reason", "")))
            return approve
        except Exception:
            approve = self.player_id in team
            self._record_reason(turn, round_num, "approved" if approve else "rejected", "fallback: approve only teams I am on")
            return approve

    async def vote_on_quest(self, turn: int, team: List[int]) -> bool:
        if self.side:
            return True  # Good must pass

        prompt = f"You are Evil on Quest {turn + 1} with team {team}. Call vote_on_quest to Pass (bluff) or Fail (score)."
        try:
            # 1. Try Tool Calling first
            args = await self._call_llm_tool(prompt, "vote_on_quest")
            if "pass_quest" in args:
                return bool(args["pass_quest"])

            # 2. Try JSON mode fallback
            resp = await self._call_llm(prompt + " Output JSON: {\"pass_quest\": true/false}", json_mode=True)
            data = json.loads(resp)
            return bool(data.get("pass_quest", False))
        except Exception:
            return False

    async def assassinate(self) -> int:
        good_players = [p for p in range(self.num_players) if p not in self.private_info.get("seen_evil", []) and p != self.player_id]
        prompt = f"Good succeeded 3 quests! You are the Assassin. Call assassinate with your target player ID among {good_players}."
        if self.memory is not None and self.memory.merlin is not None:
            prompt += " Weigh your Merlin graph and what your partner said."
        try:
            # 1. Try Tool Calling first
            args = await self._call_llm_tool(prompt, "assassinate")
            if "target" in args and int(args["target"]) in good_players:
                return int(args["target"])

            # 2. Try JSON mode fallback
            resp = await self._call_llm(prompt + " Output JSON: {\"target\": player_id}", json_mode=True)
            data = json.loads(resp)
            target = int(data.get("target"))
            if target in good_players:
                return target
        except Exception:
            pass
        guess = self.merlin_guess()
        if guess is not None:
            return guess
        return self.rng.choice(good_players) if good_players else 0
