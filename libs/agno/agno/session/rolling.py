from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from agno.models.base import Model
from agno.models.utils import get_model
from agno.run.agent import Message
from agno.utils.log import log_debug, log_warning

if TYPE_CHECKING:
    from agno.metrics import RunMetrics
    from agno.session import Session
    from agno.session.agent import AgentSession
    from agno.session.team import TeamSession


@dataclass
class RollingCompactionState:
    """Model for Rolling Session Compaction State."""

    summary: str
    compacted_through_run_id: str
    version: int
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        _dict = {
            "summary": self.summary,
            "compacted_through_run_id": self.compacted_through_run_id,
            "version": self.version,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        return {k: v for k, v in _dict.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RollingCompactionState":
        updated_at = data.get("updated_at")
        if updated_at:
            data["updated_at"] = datetime.fromisoformat(updated_at)
        return cls(**data)


class RollingCompactionResponse(BaseModel):
    """Model for Rolling Compaction Model output."""

    summary: str = Field(
        ...,
        description="The new, comprehensive summary combining the existing summary and the new conversation.",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True, indent=2)


@dataclass
class RollingCompactionManager:
    """Rolling Session Compaction Manager"""

    # Unique identifier for this manager. Auto-generated if not provided.
    id: Optional[str] = None

    # Model used for compaction generation
    model: Optional[Model] = None

    # Prompt used for session summary generation
    compaction_prompt: Optional[str] = None

    # User message prompt for requesting the summary
    compaction_request_message: str = "Update the summary with the new conversation."

    # When uncompacted runs exceed this threshold, compaction is triggered.
    compaction_run_threshold: int = 6

    # After compaction, this many runs are left uncompacted (verbatim).
    verbatim_run_budget: int = 3

    # Optional threshold in tokens to trigger compaction (not implemented yet).
    token_threshold: Optional[int] = None

    # Whether to include tool messages in the uncompacted messages passed to the model
    include_tool_messages: bool = True

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = f"rolling_compaction_manager_{uuid4().hex[:8]}"
        if self.compaction_run_threshold <= self.verbatim_run_budget:
            raise ValueError(
                f"compaction_run_threshold ({self.compaction_run_threshold}) must be greater than "
                f"verbatim_run_budget ({self.verbatim_run_budget})"
            )
        if self.verbatim_run_budget < 0:
            raise ValueError(f"verbatim_run_budget must be non-negative, got {self.verbatim_run_budget}")

    def get_response_format(self, model: "Model") -> Union[Dict[str, Any], Type[BaseModel]]:  # type: ignore
        if model.supports_native_structured_outputs:
            return RollingCompactionResponse
        elif model.supports_json_schema_outputs:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": RollingCompactionResponse.__name__,
                    "schema": RollingCompactionResponse.model_json_schema(),
                },
            }
        else:
            return {"type": "json_object"}

    def get_system_message(
        self,
        current_summary: Optional[str],
        messages_to_fold: List[Message],
        response_format: Union[Dict[str, Any], Type[BaseModel]],
    ) -> Message:
        if self.compaction_prompt is not None:
            system_prompt = self.compaction_prompt
        else:
            system_prompt = dedent("""\
            You are tasked with updating a rolling summary of a conversation between a user and an assistant.
            You will be provided with the CURRENT SUMMARY (which covers past interactions) and the NEW CONVERSATION.
            Your job is to produce a NEW SUMMARY that seamlessly integrates the new information with the existing summary.
            Keep the summary concise and focused on important information that would be helpful for future interactions.
            Do not make anything up.
            """)

        system_prompt += "\n<current_summary>\n"
        if current_summary:
            system_prompt += current_summary
        else:
            system_prompt += "None"
        system_prompt += "\n</current_summary>\n"

        conversation_messages = []
        system_prompt += "\n<new_conversation>\n"
        for message in messages_to_fold:
            if message.role == "user":
                if not message.content or (isinstance(message.content, str) and message.content.strip() == ""):
                    media_types = []
                    if hasattr(message, "images") and message.images:
                        media_types.append(f"{len(message.images)} image(s)")
                    if hasattr(message, "videos") and message.videos:
                        media_types.append(f"{len(message.videos)} video(s)")
                    if hasattr(message, "audio") and message.audio:
                        media_types.append(f"{len(message.audio)} audio file(s)")
                    if hasattr(message, "files") and message.files:
                        media_types.append(f"{len(message.files)} file(s)")
                    if media_types:
                        conversation_messages.append(f"User: [Provided {', '.join(media_types)}]")
                else:
                    conversation_messages.append(f"User: {message.content}")
            elif message.role in ["assistant", "model"]:
                conversation_messages.append(f"Assistant: {message.content}")
            elif message.role == "tool" and self.include_tool_messages:
                tool_name = message.tool_name or "unknown"
                conversation_messages.append(f"Tool ({tool_name}): {message.content}")

        system_prompt += "\n\n".join(conversation_messages)
        system_prompt += "\n</new_conversation>"

        if response_format == {"type": "json_object"}:
            from agno.utils.prompts import get_json_output_prompt

            system_prompt += "\n" + get_json_output_prompt(RollingCompactionResponse)  # type: ignore

        return Message(role="system", content=system_prompt)

    def _get_messages_to_fold(
        self, session: "Session", compacted_through_run_id: Optional[str]
    ) -> Tuple[List[Message], Optional[str]]:
        """Returns the list of messages to fold and the run_id that they go up to."""
        runs = getattr(session, "runs", [])
        if not runs:
            return [], compacted_through_run_id

        start_idx = 0
        if compacted_through_run_id:
            for idx, r in enumerate(runs):
                if r.run_id == compacted_through_run_id:
                    start_idx = idx + 1
                    break

        end_idx = len(runs) - self.verbatim_run_budget
        if start_idx >= end_idx:
            return [], compacted_through_run_id

        runs_to_fold = runs[start_idx:end_idx]
        if not runs_to_fold:
            return [], compacted_through_run_id

        new_compacted_through_run_id = runs_to_fold[-1].run_id

        messages = []
        for r in runs_to_fold:
            if r.messages:
                messages.extend(r.messages)

        return messages, new_compacted_through_run_id

    def should_compact(self, session: "Session") -> bool:
        runs = getattr(session, "runs", [])
        if not runs:
            return False

        state = getattr(session, "rolling_compaction_state", None)
        start_idx = 0
        if state and state.compacted_through_run_id:
            for idx, r in enumerate(runs):
                if r.run_id == state.compacted_through_run_id:
                    start_idx = idx + 1
                    break

        uncompacted_runs_count = len(runs) - start_idx
        return uncompacted_runs_count >= self.compaction_run_threshold

    def _process_compaction_response(
        self,
        compaction_response: Any,
        session_model: "Model",
        current_state: Optional[RollingCompactionState],
        new_run_id: str,
    ) -> Optional[RollingCompactionState]:
        if compaction_response is None:
            return None

        new_summary = None

        if (
            session_model.supports_native_structured_outputs
            and compaction_response.parsed is not None
            and isinstance(compaction_response.parsed, RollingCompactionResponse)
        ):
            new_summary = compaction_response.parsed.summary

        elif isinstance(compaction_response.content, str):
            try:
                from agno.utils.string import parse_response_model_str

                parsed: RollingCompactionResponse = parse_response_model_str(  # type: ignore
                    compaction_response.content, RollingCompactionResponse
                )
                if parsed is not None:
                    new_summary = parsed.summary
            except Exception as e:
                log_warning(f"Failed to parse rolling compaction response: {str(e)}")

        if new_summary:
            version = current_state.version + 1 if current_state else 1
            return RollingCompactionState(
                summary=new_summary,
                compacted_through_run_id=new_run_id,
                version=version,
                updated_at=datetime.now(),
            )

        return None

    def compact(
        self,
        session: Union["AgentSession", "TeamSession"],
        run_metrics: Optional["RunMetrics"] = None,
    ) -> Optional[RollingCompactionState]:
        if not self.should_compact(session):
            return getattr(session, "rolling_compaction_state", None)

        log_debug("Compacting session history", center=True)
        self.model = get_model(self.model)
        if self.model is None:
            return None

        current_state = getattr(session, "rolling_compaction_state", None)
        compacted_through = current_state.compacted_through_run_id if current_state else None
        current_summary = current_state.summary if current_state else None

        messages_to_fold, new_compacted_through_run_id = self._get_messages_to_fold(session, compacted_through)
        if not messages_to_fold or not new_compacted_through_run_id:
            log_debug("No meaningful messages to compact, skipping")
            return current_state

        response_format = self.get_response_format(self.model)
        system_message = self.get_system_message(current_summary, messages_to_fold, response_format)

        messages = [
            system_message,
            Message(role="user", content=self.compaction_request_message),
        ]

        compaction_response = self.model.response(messages=messages, response_format=response_format)

        if run_metrics is not None:
            from agno.metrics import ModelType, accumulate_model_metrics

            accumulate_model_metrics(compaction_response, self.model, ModelType.SESSION_SUMMARY_MODEL, run_metrics)

        new_state = self._process_compaction_response(
            compaction_response, self.model, current_state, new_compacted_through_run_id
        )

        if session is not None and new_state is not None:
            session.rolling_compaction_state = new_state

        return new_state

    async def acompact(
        self,
        session: Union["AgentSession", "TeamSession"],
        run_metrics: Optional["RunMetrics"] = None,
    ) -> Optional[RollingCompactionState]:
        if not self.should_compact(session):
            return getattr(session, "rolling_compaction_state", None)

        log_debug("Compacting session history (async)", center=True)
        self.model = get_model(self.model)
        if self.model is None:
            return None

        current_state = getattr(session, "rolling_compaction_state", None)
        compacted_through = current_state.compacted_through_run_id if current_state else None
        current_summary = current_state.summary if current_state else None

        messages_to_fold, new_compacted_through_run_id = self._get_messages_to_fold(session, compacted_through)
        if not messages_to_fold or not new_compacted_through_run_id:
            log_debug("No meaningful messages to compact, skipping")
            return current_state

        response_format = self.get_response_format(self.model)
        system_message = self.get_system_message(current_summary, messages_to_fold, response_format)

        messages = [
            system_message,
            Message(role="user", content=self.compaction_request_message),
        ]

        compaction_response = await self.model.aresponse(messages=messages, response_format=response_format)

        if run_metrics is not None:
            from agno.metrics import ModelType, accumulate_model_metrics

            accumulate_model_metrics(compaction_response, self.model, ModelType.SESSION_SUMMARY_MODEL, run_metrics)

        new_state = self._process_compaction_response(
            compaction_response, self.model, current_state, new_compacted_through_run_id
        )

        if session is not None and new_state is not None:
            session.rolling_compaction_state = new_state

        return new_state
