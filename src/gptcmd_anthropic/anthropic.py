"""
This module contains an implementation of the Gptcmd LLMProvider for
Anthropic's models.
Copyright 2024 Bill Dengler
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import inspect

from copy import deepcopy
from decimal import Decimal
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import anthropic

from gptcmd.llm import (
    CompletionError,
    InvalidAPIParameterError,
    LLMProvider,
    LLMProviderFeature,
    LLMResponse,
)
from gptcmd.message import Image, Message, MessageRole


class _CacheTarget(NamedTuple):
    index: int
    block_end: int
    weight: int


class _CacheSelection(NamedTuple):
    to_cache: Set[int]
    request_cache_active: bool


class AnthropicProvider(LLMProvider):
    SUPPORTED_FEATURES = LLMProviderFeature.RESPONSE_STREAMING
    _DEFAULT_CACHE_CONTROL: Dict[str, Any] = {"type": "ephemeral"}
    _MAX_CACHE_BREAKPOINTS = 4
    _MAX_CACHE_LOOKBACK_BLOCKS = 20
    # Weight is num_blocks * 1000 + total text chars; 8000 is roughly a
    # single-block message with 7000 text characters.
    _MIN_LARGE_CACHE_MESSAGE_WEIGHT = 8000

    def __init__(self, client, *args, **kwargs):
        self._anthropic = client
        self._models = {m.id for m in self._anthropic.models.list()}
        super().__init__(*args, **kwargs)
        self._stream = True

    @classmethod
    def from_config(cls, conf: Dict):
        special_opts = (
            "model",
            "provider",
        )
        model = conf.get("model")
        client_opts = {k: v for k, v in conf.items() if k not in special_opts}
        client = anthropic.Anthropic(**client_opts)
        return cls(client, model=model)

    @staticmethod
    def _estimate_cost_in_cents(
        model: str,
        prompt_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
        sampled_tokens: int,
    ) -> Optional[Decimal]:
        COST_PER_PROMPT_SAMPLED: Dict[str, Tuple[Decimal, Decimal]] = {
            "claude-mythos-5": (
                Decimal("10") / Decimal("1000000"),
                Decimal("50") / Decimal("1000000"),
            ),
            "claude-fable-5": (
                Decimal("10") / Decimal("1000000"),
                Decimal("50") / Decimal("1000000"),
            ),
            "claude-opus-4-8": (
                Decimal("5") / Decimal("1000000"),
                Decimal("25") / Decimal("1000000"),
            ),
            "claude-opus-4-7": (
                Decimal("5") / Decimal("1000000"),
                Decimal("25") / Decimal("1000000"),
            ),
            "claude-opus-4-6": (
                Decimal("5") / Decimal("1000000"),
                Decimal("25") / Decimal("1000000"),
            ),
            "claude-opus-4-5-20251101": (
                Decimal("5") / Decimal("1000000"),
                Decimal("25") / Decimal("1000000"),
            ),
            "claude-sonnet-4-6": (
                Decimal("3") / Decimal("1000000"),
                Decimal("15") / Decimal("1000000"),
            ),
            "claude-sonnet-4-5-20250929": (
                Decimal("3") / Decimal("1000000"),
                Decimal("15") / Decimal("1000000"),
            ),
            "claude-haiku-4-5-20251001": (
                Decimal("1") / Decimal("1000000"),
                Decimal("5") / Decimal("1000000"),
            ),
            "claude-opus-4-1-20250805": (
                Decimal("15") / Decimal("1000000"),
                Decimal("75") / Decimal("1000000"),
            ),
            "claude-sonnet-4-20250514": (
                Decimal("3") / Decimal("1000000"),
                Decimal("15") / Decimal("1000000"),
            ),
            "claude-3-7-sonnet-20250219": (
                Decimal("3") / Decimal("1000000"),
                Decimal("15") / Decimal("1000000"),
            ),
            "claude-3-5-sonnet-20241022": (
                Decimal("3") / Decimal("1000000"),
                Decimal("15") / Decimal("1000000"),
            ),
            "claude-3-5-haiku-20241022": (
                Decimal("1") / Decimal("1000000"),
                Decimal("5") / Decimal("1000000"),
            ),
            "claude-opus-4-20250514": (
                Decimal("15") / Decimal("1000000"),
                Decimal("75") / Decimal("1000000"),
            ),
            "claude-3-opus-20240229": (
                Decimal("15") / Decimal("1000000"),
                Decimal("75") / Decimal("1000000"),
            ),
            "claude-3-sonnet-20240229": (
                Decimal("3") / Decimal("1000000"),
                Decimal("15") / Decimal("1000000"),
            ),
            "claude-3-haiku-20240307": (
                Decimal("0.25") / Decimal("1000000"),
                Decimal("1.25") / Decimal("1000000"),
            ),
        }

        CACHE_WRITE_MULTIPLIER: Decimal = Decimal("1.25")
        CACHE_READ_MULTIPLIER: Decimal = Decimal("0.1")

        if model not in COST_PER_PROMPT_SAMPLED:
            return None

        prompt_scale, sampled_scale = COST_PER_PROMPT_SAMPLED[model]
        cache_write_scale = prompt_scale * CACHE_WRITE_MULTIPLIER
        cache_read_scale = prompt_scale * CACHE_READ_MULTIPLIER

        return (
            Decimal(prompt_tokens) * prompt_scale
            + Decimal(cache_write_tokens) * cache_write_scale
            + Decimal(cache_read_tokens) * cache_read_scale
            + Decimal(sampled_tokens) * sampled_scale
        ) * Decimal("100")

    @staticmethod
    def _max_tokens_cap(model: str) -> int:
        """Return the model-specific hard limit, else 4096."""
        by_model = {
            "claude-mythos-5": 128000,
            "claude-fable-5": 128000,
            "claude-opus-4-8": 128000,
            "claude-opus-4-7": 128000,
            "claude-opus-4-6": 128000,
            "claude-opus-4-5-20251101": 64000,
            "claude-opus-4-1-20250805": 32000,
            "claude-opus-4-20250514": 32000,
            "claude-sonnet-4-6": 64000,
            "claude-sonnet-4-5-20250929": 64000,
            "claude-sonnet-4-20250514": 64000,
            "claude-3-7-sonnet-20250219": 64000,
            "claude-3-5-sonnet-20241022": 8192,
            "claude-3-5-sonnet-20240620": 8192,
            "claude-haiku-4-5-20251001": 64000,
            "claude-3-5-haiku-20241022": 8192,
        }
        return by_model.get(model, 4096)

    @staticmethod
    def _collapse_content(
        content1: Any, content2: Any
    ) -> List[Dict[str, Any]]:
        if isinstance(content1, str):
            content1 = [{"type": "text", "text": content1}]
        if isinstance(content2, str):
            content2 = [{"type": "text", "text": content2}]
        if not isinstance(content1, list) or not isinstance(content2, list):
            raise CompletionError("Unexpected Anthropic content types")
        return deepcopy(content1) + deepcopy(content2)

    @staticmethod
    def _message_weight(content: Sequence[Dict[str, Any]]) -> int:
        total_text_length = sum(
            len(block.get("text", ""))
            for block in content
            if block.get("type") == "text"
        )
        return len(content) * 1000 + total_text_length

    def _should_cache_system(
        self,
        system_text: str,
        force_cache_system: Optional[bool],
        always_cache: Set[int],
    ) -> bool:
        if not system_text:
            return False

        if force_cache_system is not None:
            return bool(force_cache_system)

        return len(always_cache) < self._MAX_CACHE_BREAKPOINTS

    @classmethod
    def _select_spaced_targets(
        cls,
        candidates: Sequence[_CacheTarget],
        slots: int,
        reserved_positions: Sequence[int],
    ) -> List[_CacheTarget]:
        """Choose up to ``slots`` evenly spaced targets. Pure."""
        if slots <= 0 or not candidates:
            return []
        reserved = list(reserved_positions)
        ordered = sorted(candidates, key=lambda t: t.block_end)
        usable_slots = min(slots, len(ordered))
        start_block = ordered[0].block_end
        end_block = ordered[-1].block_end
        # Half the API's 20-block cache lookback keeps explicit markers
        # spaced enough that each lookup has room to find earlier entries.
        min_gap = cls._MAX_CACHE_LOOKBACK_BLOCKS // 2
        selected: List[_CacheTarget] = []
        selected_indices: Set[int] = set()
        for step in range(1, usable_slots + 1):
            target_block = start_block + (
                (end_block - start_block) * step / (usable_slots + 1)
            )
            available = [t for t in ordered if t.index not in selected_indices]
            if not available:
                break
            chosen = min(
                available,
                key=lambda t: (
                    int(
                        min(
                            (abs(t.block_end - pos) for pos in reserved),
                            default=min_gap,
                        )
                        < min_gap
                    ),
                    abs(t.block_end - target_block),
                    -t.weight,
                    t.block_end,
                ),
            )
            selected.append(chosen)
            selected_indices.add(chosen.index)
            reserved.append(chosen.block_end)
        return selected

    def _select_default_cache_targets(
        self,
        anthropic_messages: Sequence[Dict[str, Any]],
        system_text: str,
        always_cache: Set[int],
        never_cache: Set[int],
        should_cache_system: bool,
        *,
        force_request_cache: bool = False,
        prefix_block_count: int = 0,
    ) -> _CacheSelection:
        """Select automatic explicit and request-level cache breakpoints.

        Block numbering is 1-indexed across tools, then system, then
        messages. The automatic request-level breakpoint consumes one of the
        four cache slots when active. Explicit slots go first to forced
        metadata breakpoints and the system block, then to one large message,
        then to evenly spaced stable-region messages kept at least ten blocks
        from existing breakpoints where possible. When request-level caching
        is active or the last message is cached by the metadata field,
        prefer candidates whose block_end falls before the final 20-block
        cache lookback window.
        """
        targets: List[_CacheTarget] = []
        next_block_start = prefix_block_count + (2 if system_text else 1)
        for i, msg in enumerate(anthropic_messages):
            block_count = len(msg["content"])
            if block_count <= 0:
                continue
            block_end = next_block_start + block_count - 1
            targets.append(
                _CacheTarget(
                    index=i,
                    block_end=block_end,
                    weight=self._message_weight(msg["content"]),
                )
            )
            next_block_start = block_end + 1

        total_blocks = next_block_start - 1
        mandatory_breakpoints = len(always_cache) + int(should_cache_system)
        tail_always_cache = bool(targets) and targets[-1].index in always_cache
        tail_never_cache = (
            bool(targets)
            and targets[-1].index in never_cache
            and targets[-1].index not in always_cache
        )
        request_cache_active = force_request_cache or (
            total_blocks > 0
            and mandatory_breakpoints < self._MAX_CACHE_BREAKPOINTS
            and not tail_always_cache
            and not tail_never_cache
        )
        explicit_budget = self._MAX_CACHE_BREAKPOINTS - int(
            request_cache_active
        )
        free_explicit_slots = max(
            explicit_budget - mandatory_breakpoints,
            0,
        )

        reserved_positions: List[int] = []
        if should_cache_system:
            reserved_positions.append(prefix_block_count + 1)

        target_lookup = {target.index: target for target in targets}
        reserved_positions.extend(
            target_lookup[i].block_end
            for i in sorted(always_cache)
            if i in target_lookup
        )

        auto_to_cache: Set[int] = set()
        eligible = [
            target
            for target in targets
            if target.index not in always_cache
            and target.index not in never_cache
        ]
        explicit_candidates = eligible
        if request_cache_active and targets:
            tail_index = targets[-1].index
            explicit_candidates = [
                target for target in eligible if target.index != tail_index
            ]

        if free_explicit_slots > 0 and explicit_candidates:
            remaining_slots = free_explicit_slots
            large_message_candidates = [
                target
                for target in explicit_candidates
                if (
                    target.index not in auto_to_cache
                    and target.weight >= self._MIN_LARGE_CACHE_MESSAGE_WEIGHT
                )
            ]

            if large_message_candidates:
                large_message = min(
                    large_message_candidates,
                    key=lambda target: (
                        -target.weight,
                        target.block_end,
                    ),
                )
                auto_to_cache.add(large_message.index)
                reserved_positions.append(large_message.block_end)
                remaining_slots -= 1

            if remaining_slots > 0:
                preferred_candidates = [
                    target
                    for target in explicit_candidates
                    if target.index not in auto_to_cache
                ]
                if request_cache_active or tail_always_cache:
                    tail_start = max(
                        1,
                        total_blocks - self._MAX_CACHE_LOOKBACK_BLOCKS + 1,
                    )
                    preferred_candidates = [
                        target
                        for target in preferred_candidates
                        if target.block_end < tail_start
                    ]
                    if not preferred_candidates:
                        preferred_candidates = [
                            target
                            for target in explicit_candidates
                            if target.index not in auto_to_cache
                        ]

                selected = self._select_spaced_targets(
                    preferred_candidates,
                    remaining_slots,
                    reserved_positions,
                )
                auto_to_cache.update(target.index for target in selected)
                reserved_positions.extend(
                    target.block_end for target in selected
                )
                remaining_slots -= len(selected)

            remaining_candidates = [
                target
                for target in explicit_candidates
                if target.index not in auto_to_cache
            ]
            if remaining_slots > 0 and remaining_candidates:
                selected = self._select_spaced_targets(
                    remaining_candidates,
                    remaining_slots,
                    reserved_positions,
                )
                auto_to_cache.update(target.index for target in selected)
                reserved_positions.extend(
                    target.block_end for target in selected
                )
                remaining_slots -= len(selected)

        return _CacheSelection(
            to_cache=always_cache | auto_to_cache,
            request_cache_active=request_cache_active,
        )

    @staticmethod
    def _apply_message_cache_control(
        anthropic_messages: Sequence[Dict[str, Any]],
        to_cache: Set[int],
        cache_control: Dict[str, Any],
    ) -> None:
        for idx in to_cache:
            msg = anthropic_messages[idx]
            block = next(
                (
                    b
                    for b in reversed(msg["content"])
                    if b["type"] not in ("thinking", "redacted_thinking")
                ),
                None,
            )
            if block is not None:
                block["cache_control"] = cache_control.copy()

    @classmethod
    def _build_system_blocks(
        cls,
        system_text: str,
        should_cache_system: bool,
        cache_control: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        if not system_text:
            return None
        system_block = {"type": "text", "text": system_text}
        if should_cache_system:
            if cache_control is None:
                cache_control = cls._DEFAULT_CACHE_CONTROL
            system_block["cache_control"] = cache_control.copy()
        return [system_block]

    @staticmethod
    def _extract_usage(
        resp: Any,
    ) -> Tuple[Optional[int], int, int, Optional[int]]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return (None, 0, 0, None)
        return (
            getattr(usage, "input_tokens", None),
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "output_tokens", None),
        )

    @staticmethod
    def _clamp(val, bottom, top):
        return max(min(val, top), bottom)

    @classmethod
    def _append_message(
        cls,
        messages: List[Dict[str, Any]],
        rendered_message: Dict[str, Any],
    ) -> int:
        if messages and messages[-1]["role"] == rendered_message["role"]:
            messages[-1]["content"] = cls._collapse_content(
                messages[-1]["content"],
                rendered_message["content"],
            )
            return len(messages) - 1
        messages.append(rendered_message)
        return len(messages) - 1

    @staticmethod
    def _render_assistant_message(role: str, blocks: Sequence[Any]) -> Message:
        output = []
        thinking_text = []
        thinking_signature = []

        for block in blocks:
            if block.type == "text":
                output.append(block.text)
            elif block.type == "thinking":
                if hasattr(block, "thinking"):
                    thinking_text.append(block.thinking)
                if hasattr(block, "signature"):
                    thinking_signature.append(block.signature)

        msg = Message(content="".join(output), role=role)
        if thinking_text or thinking_signature:
            meta = msg.metadata
            if thinking_text:
                meta["anthropic_thinking_text"] = "".join(thinking_text)
            if thinking_signature:
                meta["anthropic_thinking_signature"] = "".join(
                    thinking_signature
                )
        return msg

    @classmethod
    def _set_response_usage(
        cls,
        response: LLMResponse,
        model: str,
        usage: Tuple[Optional[int], int, int, Optional[int]],
    ) -> None:
        input_tokens, cache_write, cache_read, output_tokens = usage
        response.sampled_tokens = output_tokens
        if input_tokens is None:
            response.prompt_tokens = None
            response.cost_in_cents = None
            return

        response.prompt_tokens = input_tokens + cache_write
        if output_tokens is None:
            response.cost_in_cents = None
            return

        response.cost_in_cents = cls._estimate_cost_in_cents(
            model=model,
            prompt_tokens=input_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            sampled_tokens=output_tokens,
        )

    def _message_content_to_anthropic(
        self, msg: Message
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if (
            "anthropic_thinking_text" in msg.metadata
            and "anthropic_thinking_signature" in msg.metadata
        ):
            content.append(
                {
                    "type": "thinking",
                    "signature": msg.metadata["anthropic_thinking_signature"],
                    "thinking": msg.metadata["anthropic_thinking_text"],
                }
            )
        content.extend(
            [
                {"type": "text", "text": msg.content},
                *[self.format_attachment(a) for a in msg.attachments],
            ]
        )
        return content

    def _message_to_anthropic(self, msg: Message) -> Dict[str, Any]:
        return {
            "role": msg.role,
            "content": self._message_content_to_anthropic(msg),
        }

    def _collect_request_messages(
        self, messages: Sequence[Message]
    ) -> Tuple[
        List[Dict[str, Any]],
        str,
        Optional[bool],
        Set[int],
        Set[int],
    ]:
        anthropic_messages: List[Dict[str, Any]] = []
        system_text = ""
        force_cache_system = None
        always_cache: Set[int] = set()
        never_cache: Set[int] = set()

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                if msg.attachments:
                    raise CompletionError(
                        "Attachments on system messages aren't supported"
                    )
                if system_text:
                    system_text += "\n\n" + msg.content
                else:
                    system_text = msg.content

                val = msg.metadata.get("anthropic_cache_breakpoint")
                if val is True:
                    force_cache_system = True
                elif val is False and force_cache_system is not True:
                    force_cache_system = False
                continue

            target_idx = self._append_message(
                anthropic_messages,
                self._message_to_anthropic(msg),
            )

            # Claude rejects consecutive messages with the same role, so
            # metadata needs to follow the rendered span rather than the
            # original message index.
            val = msg.metadata.get("anthropic_cache_breakpoint")
            if val is True:
                always_cache.add(target_idx)
            elif val is False:
                never_cache.add(target_idx)

        return (
            anthropic_messages,
            system_text,
            force_cache_system,
            always_cache,
            never_cache,
        )

    def complete(self, messages: Sequence[Message]) -> LLMResponse:
        api_params = self.api_params.copy()
        user_cache_control = api_params.pop("cache_control", None)
        kwargs = {
            "model": self.model,
            "stream": self.stream,
            **api_params,
        }
        kwargs.setdefault("max_tokens", self._max_tokens_cap(self.model))
        (
            kwargs["messages"],
            system_text,
            force_cache_system,
            always_cache,
            never_cache,
        ) = self._collect_request_messages(messages)
        always_cache = {
            i
            for i in always_cache
            if i < len(kwargs["messages"]) and kwargs["messages"][i]["content"]
        }

        should_cache_system = self._should_cache_system(
            system_text,
            force_cache_system,
            always_cache,
        )
        extra_body = kwargs.get("extra_body")
        extra_body_cache_control = None
        if isinstance(extra_body, dict):
            extra_body_cache_control = extra_body.get("cache_control")
        user_top_level_cache = extra_body_cache_control is not None

        explicit_breakpoints = len(always_cache) + int(should_cache_system)
        if (
            user_top_level_cache
            and explicit_breakpoints >= self._MAX_CACHE_BREAKPOINTS
        ):
            raise CompletionError(
                "Anthropic does not allow a request-level cache_control "
                f"together with {self._MAX_CACHE_BREAKPOINTS} explicit "
                "cache breakpoints; remove cache_control from extra_body "
                "or unset anthropic_cache_breakpoint metadata on a "
                "message."
            )
        if explicit_breakpoints > self._MAX_CACHE_BREAKPOINTS:
            raise CompletionError(
                f"Anthropic supports at most {self._MAX_CACHE_BREAKPOINTS}"
                " explicit cache breakpoints"
            )
        tools = kwargs.get("tools")
        if isinstance(tools, (list, tuple)):
            tool_block_count = len(tools)
        else:
            tool_block_count = 0
        cache_selection = self._select_default_cache_targets(
            kwargs["messages"],
            system_text,
            always_cache,
            never_cache,
            should_cache_system,
            force_request_cache=user_top_level_cache,
            prefix_block_count=tool_block_count,
        )
        to_cache = cache_selection.to_cache
        request_cache_active = cache_selection.request_cache_active

        effective_cache_control = self._DEFAULT_CACHE_CONTROL.copy()
        if user_cache_control is not None:
            effective_cache_control.update(user_cache_control)
        if extra_body_cache_control is not None:
            effective_cache_control.update(extra_body_cache_control)
        # One shared cache_control guarantees one TTL per request. This avoids
        # the API's 400 when the last explicit breakpoint TTL differs from
        # the automatic request-level TTL, and sidesteps the rule that longer
        # TTL breakpoints must precede shorter-TTL breakpoints.
        self._apply_message_cache_control(
            kwargs["messages"], to_cache, effective_cache_control
        )

        system = self._build_system_blocks(
            system_text,
            should_cache_system,
            effective_cache_control if should_cache_system else None,
        )
        if system is not None:
            kwargs["system"] = system

        if request_cache_active:
            extra_body = kwargs.get("extra_body")
            if extra_body is None:
                extra_body = {}
            else:
                extra_body = extra_body.copy()

            extra_body["cache_control"] = effective_cache_control.copy()
            kwargs["extra_body"] = extra_body

        try:
            resp = self._anthropic.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise CompletionError(str(e)) from e

        if isinstance(resp, anthropic.Stream):
            return StreamedClaudeResponse(resp, self)

        if resp.stop_reason == "refusal":
            stop_details = getattr(resp, "stop_details", None)
            if isinstance(stop_details, dict):
                explanation = stop_details.get("explanation")
            else:
                explanation = getattr(stop_details, "explanation", None)
            raise CompletionError(
                explanation or "Anthropic refused the request"
            )

        res = LLMResponse(
            message=self._render_assistant_message(resp.role, resp.content)
        )
        self._set_response_usage(res, resp.model, self._extract_usage(resp))
        return res

    def get_best_model(self):
        return "claude-opus-4-8"

    @property
    def valid_models(self):
        return self._models | {
            # Some model aliases aren't included in the API-provided list.
            # Include these manually.
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
        }

    def validate_api_params(self, params):
        special_opts = frozenset(("model", "messages", "stream", "system"))
        supported_extra_opts = frozenset(("cache_control", "extra_body"))
        valid_opts = (
            frozenset(
                inspect.signature(
                    self._anthropic.messages.create
                ).parameters.keys()
            )
            - special_opts
            | supported_extra_opts
        )
        clamped = {"temperature": (0, 1)}

        for opt in params:
            if opt not in valid_opts:
                raise InvalidAPIParameterError(f"Unknown parameter {opt}")
            if opt == "cache_control":
                if params[opt] is None:
                    continue
                if not isinstance(params[opt], dict):
                    raise InvalidAPIParameterError(
                        "cache_control must be a dict"
                    )
            elif opt == "max_tokens":
                hi = self.__class__._max_tokens_cap(self.model)
                params[opt] = self.__class__._clamp(params[opt], 0, hi)
            elif opt == "extra_body":
                extra_body = params[opt]
                if extra_body is None:
                    continue
                if not isinstance(extra_body, dict):
                    raise InvalidAPIParameterError("extra_body must be a dict")
                extra_body_cache_control = extra_body.get("cache_control")
                if extra_body_cache_control is None:
                    continue
                if not isinstance(extra_body_cache_control, dict):
                    raise InvalidAPIParameterError(
                        "extra_body cache_control must be a dict"
                    )
            elif opt in clamped:
                params[opt] = self.__class__._clamp(params[opt], *clamped[opt])
        return params


class StreamedClaudeResponse(LLMResponse):
    def __init__(self, backing_stream, provider: AnthropicProvider):
        self._stream = backing_stream
        self._provider = provider
        self._model: str = ""
        self._prompt = 0
        self._cache_write = 0
        self._cache_read = 0
        self._sampled = 0

        message = Message(content="", role="")
        super().__init__(message)

    def _put_metadata(self, key: str, addition: str) -> None:
        meta = self.message.metadata
        meta[key] = meta.get(key, "") + addition

    def _update_usage(self, usage_obj: Any) -> None:
        self._prompt += getattr(usage_obj, "input_tokens", 0) or 0
        self._cache_write += (
            getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
        )
        self._cache_read += (
            getattr(usage_obj, "cache_read_input_tokens", 0) or 0
        )
        self._sampled += getattr(usage_obj, "output_tokens", 0) or 0

    def __iter__(self):
        try:
            for chunk in self._stream:
                if hasattr(chunk, "usage"):
                    self._update_usage(chunk.usage)
                    self.prompt_tokens = self._prompt + self._cache_write
                    self.sampled_tokens = self._sampled
                    if self._model:
                        self.cost_in_cents = (
                            self._provider.__class__._estimate_cost_in_cents(
                                model=self._model,
                                prompt_tokens=self.prompt_tokens,
                                cache_write_tokens=self._cache_write,
                                cache_read_tokens=self._cache_read,
                                sampled_tokens=self.sampled_tokens,
                            )
                        )
                if chunk.type == "message_start":
                    if hasattr(chunk.message, "model"):
                        self._model = chunk.message.model
                    if hasattr(chunk.message, "role"):
                        self.message.role = chunk.message.role
                    if hasattr(chunk.message, "usage"):
                        self._update_usage(chunk.message.usage)
                elif (
                    chunk.type == "content_block_delta"
                    and chunk.delta.type == "text_delta"
                ):
                    next_text = chunk.delta.text
                    self.message.content += next_text
                    yield next_text
                elif (
                    chunk.type == "content_block_delta"
                    and chunk.delta.type == "thinking_delta"
                ):
                    self._put_metadata(
                        "anthropic_thinking_text", chunk.delta.thinking
                    )
                elif (
                    chunk.type == "content_block_delta"
                    and chunk.delta.type == "signature_delta"
                ):
                    self._put_metadata(
                        "anthropic_thinking_signature", chunk.delta.signature
                    )
                elif chunk.type == "message_delta":
                    if chunk.delta.stop_reason == "refusal":
                        stop_details = getattr(
                            chunk.delta, "stop_details", None
                        )
                        if isinstance(stop_details, dict):
                            explanation = stop_details.get("explanation")
                        else:
                            explanation = getattr(
                                stop_details, "explanation", None
                            )
                        raise CompletionError(
                            explanation or "Anthropic refused the request"
                        )
        except anthropic.APIError as e:
            raise CompletionError(str(e)) from e


@AnthropicProvider.register_attachment_formatter(Image)
def format_image_for_claude(img):
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": img.mimetype,
            "data": img.b64,
        },
    }
