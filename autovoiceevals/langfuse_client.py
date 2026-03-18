"""Langfuse client for prompt management, evaluation tracing, and scoring.

Prompt management:
  - get_system_prompt()  -> fetch versioned prompt from Langfuse
  - update_prompt()      -> create new prompt version in Langfuse

Single-turn evaluation:
  - run_single_turn()    -> send context to Gemini, trace in Langfuse

Multi-turn (legacy):
  - run_conversation()   -> Claude-based conversation simulation

Datasets & Scoring:
  - upload_dataset()     -> store eval items as Langfuse dataset items
  - score_trace()        -> report evaluation scores to Langfuse
"""

from __future__ import annotations

import time
from datetime import datetime

from langfuse import Langfuse

from .models import Turn, Conversation, DatasetItem

END_PHRASES = [
    "have a great day", "goodbye", "talk to you soon", "take care",
]


class LangfuseClient:
    """Client using Langfuse for prompt management and tracing."""

    def __init__(
        self,
        langfuse_public_key: str,
        langfuse_secret_key: str,
        langfuse_host: str,
        llm_client=None,
        prompt_name: str = "Assistant System Prompt - USA - Named",
    ):
        self.langfuse = Langfuse(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            host=langfuse_host,
        )
        self.llm = llm_client
        self.prompt_name = prompt_name

        # Tracks the last trace for external scoring
        self._last_trace = None
        self._dataset_name = None

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------

    def get_system_prompt(self, assistant_id: str) -> str:
        """Fetch the latest prompt from Langfuse.

        Args:
            assistant_id: Used as the {{NAME}} template variable value.

        Handles both text and chat prompt types. For chat prompts,
        extracts the system message content.
        """
        prompt = self.langfuse.get_prompt(
            name=self.prompt_name,
            label="latest",
        )
        compiled = prompt.compile(NAME=assistant_id)

        # Chat prompts return a list of message dicts
        if isinstance(compiled, list):
            system_msgs = [m for m in compiled if m.get("role") == "system"]
            if system_msgs:
                return system_msgs[0]["content"]
            # Fallback: concatenate all message contents
            return "\n".join(m.get("content", "") for m in compiled)

        # Text prompts return a string directly
        return compiled

    def update_prompt(self, assistant_id: str, new_prompt: str) -> bool:
        """Create a new prompt version in Langfuse.

        Each call auto-increments the version. The 'latest' label is
        moved to the new version so get_system_prompt() always fetches it.

        Creates a chat-type prompt (matching the existing format) with
        the system message content updated and {{NAME}} as a template variable.
        """
        try:
            self.langfuse.create_prompt(
                name=self.prompt_name,
                prompt=[
                    {"role": "system", "content": new_prompt},
                    {"role": "user", "content": "{{input}}"},
                ],
                labels=["latest"],
                type="chat",
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Single-turn evaluation (Gemini-based, with Langfuse tracing)
    # ------------------------------------------------------------------

    def run_single_turn(
        self,
        system_prompt: str,
        item: DatasetItem,
        gemini_client,
    ) -> str:
        """Send conversation context to Gemini, get ONE response, trace in Langfuse.

        Args:
            system_prompt: The voice agent's system prompt.
            item: The dataset item with conversation context.
            gemini_client: GeminiClient instance for calling the production model.

        Returns:
            The agent's response text.
        """
        session_id = f"eval-{datetime.now().strftime('%Y%m%d-%H%M')}-{item.id}"

        # Start a Langfuse trace
        trace = self.langfuse.start_span(
            name=f"single-turn-{item.id}",
            input={
                "item_id": item.id,
                "description": item.description,
                "category": item.category,
                "scenario_type": item.scenario_type,
                "context_turns": len(item.conversation_context),
            },
            metadata={"session_id": session_id},
        )
        trace.update_trace(session_id=session_id, name=f"eval-{item.id}")

        try:
            t0 = time.time()
            response = gemini_client.generate(
                system_prompt=system_prompt,
                conversation_context=item.conversation_context,
            )
            latency_ms = (time.time() - t0) * 1000
        except Exception as e:
            trace.update(output={"error": str(e)[:200]}, level="ERROR")
            trace.end()
            self._last_trace = trace
            return f"[ERROR: {str(e)[:100]}]"

        # Record the Gemini call as a generation observation
        gen = trace.start_generation(
            name="gemini-response",
            model=gemini_client.model,
            input={
                "system_prompt_len": len(system_prompt),
                "context": item.conversation_context,
            },
            output=response,
            metadata={"latency_ms": round(latency_ms, 1)},
        )
        gen.end()

        # End trace with the response
        trace.update(output={"agent_response": response})
        trace.end()

        self._last_trace = trace
        return response

    # ------------------------------------------------------------------
    # Multi-turn conversation (legacy, Claude-based)
    # ------------------------------------------------------------------

    def run_conversation(
        self,
        assistant_id: str,
        scenario_id: str,
        caller_turns: list[str],
        max_turns: int = 12,
    ) -> Conversation:
        """Run a simulated multi-turn conversation with Langfuse tracing.

        Legacy method — kept for backward compatibility with pipeline mode.
        Uses Claude as the responding model.
        """
        if self.llm is None:
            raise RuntimeError(
                "LangfuseClient requires an LLMClient for conversations. "
                "Pass llm_client= to the constructor."
            )

        conv = Conversation(scenario_id=scenario_id)
        total_latency = 0.0
        session_id = f"eval-{datetime.now().strftime('%Y%m%d-%H%M')}-{scenario_id}"

        try:
            system_prompt = self.get_system_prompt(assistant_id)
        except Exception as e:
            conv.error = f"Failed to fetch prompt: {str(e)[:200]}"
            return conv

        trace = self.langfuse.start_span(
            name=f"conversation-{scenario_id}",
            input={"scenario_id": scenario_id, "system_prompt_len": len(system_prompt)},
            metadata={"assistant_id": assistant_id, "session_id": session_id},
        )
        trace.update_trace(session_id=session_id, name=f"eval-{scenario_id}")

        messages: list[dict] = []

        for turn_num, msg in enumerate(caller_turns[:max_turns], 1):
            if not msg or not msg.strip():
                msg = "..."

            conv.turns.append(Turn(role="caller", content=msg))
            messages.append({"role": "user", "content": msg})

            try:
                t0 = time.time()
                agent_response = self.llm.chat(
                    system=system_prompt,
                    messages=messages,
                    max_tokens=500,
                )
                latency = (time.time() - t0) * 1000
            except Exception as e:
                conv.error = f"LLM error: {str(e)[:200]}"
                break

            gen = trace.start_generation(
                name=f"turn-{turn_num}",
                model=self.llm.model,
                input=messages.copy(),
                output=agent_response,
                metadata={"latency_ms": round(latency, 1), "turn": turn_num},
            )
            gen.end()

            conv.turns.append(
                Turn(role="assistant", content=agent_response, latency_ms=latency)
            )
            messages.append({"role": "assistant", "content": agent_response})
            total_latency += latency

            if any(p in agent_response.lower() for p in END_PHRASES):
                break

            time.sleep(0.1)

        n = len(conv.agent_turns)
        conv.avg_latency_ms = total_latency / n if n else 0

        trace.update(output={"transcript": conv.transcript, "num_turns": len(conv.turns)})
        trace.end()

        self._last_trace = trace
        return conv

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def upload_dataset(self, dataset_name: str, items) -> None:
        """Upload eval items as Langfuse dataset items.

        Accepts both DatasetItem and Scenario objects.
        """
        self.langfuse.create_dataset(name=dataset_name)
        self._dataset_name = dataset_name

        for item in items:
            # DatasetItem (single-turn)
            if hasattr(item, 'conversation_context'):
                self.langfuse.create_dataset_item(
                    dataset_name=dataset_name,
                    input=item.to_dict(),
                    metadata={
                        "category": item.category,
                        "scenario_type": item.scenario_type,
                        "difficulty": item.difficulty,
                        "item_id": item.id,
                    },
                )
            # Scenario (legacy)
            else:
                self.langfuse.create_dataset_item(
                    dataset_name=dataset_name,
                    input=item.to_dict(),
                    metadata={
                        "persona": item.persona_name,
                        "difficulty": item.difficulty,
                        "scenario_id": item.id,
                    },
                )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_trace(self, eval_result) -> None:
        """Report evaluation scores to Langfuse on the last trace."""
        if self._last_trace is None:
            return

        self._last_trace.score_trace(name="composite_score", value=eval_result.score)
        self._last_trace.score_trace(name="should_score", value=eval_result.should_score)
        self._last_trace.score_trace(name="should_not_score", value=eval_result.should_not_score)
        self._last_trace.score_trace(name="csat", value=float(eval_result.csat_score))
        self._last_trace.score_trace(
            name="passed",
            value="true" if eval_result.passed else "false",
            data_type="CATEGORICAL",
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush all pending Langfuse events."""
        self.langfuse.flush()
