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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import anthropic

from gptcmd.llm import (
    CompletionError,
    InvalidAPIParameterError,
    LLMProvider,
    LLMProviderFeature,
    LLMResponse,
)
from gptcmd.message import Image, Message, MessageRole


class AnthropicProvider(LLMProvider):
    SUPPORTED_FEATURES = LLMProviderFeature.RESPONSE_STREAMING

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

    @staticmethod
    def _should_cache_system(
        system_text: str, force_cache_system: Optional[bool]
    ) -> bool:
        if not system_text:
            return False
        if force_cache_system is True:
            return True
        if force_cache_system is False:
            return False
        return True

    @classmethod
    def _select_default_cache_targets(
        cls,
        anthropic_messages: Sequence[Dict[str, Any]],
        always_cache: Set[int],
        never_cache: Set[int],
        should_cache_system: bool,
    ) -> Set[int]:
        cache_weights = [
            (i, cls._message_weight(msg["content"]))
            for i, msg in enumerate(anthropic_messages)
        ]
        cache_candidates = sorted(
            cache_weights, key=lambda item: item[1], reverse=True
        )

        max_cache_breakpoints = 4
        cache_slots = max(
            (
                max_cache_breakpoints
                - len(always_cache)
                - (1 if should_cache_system else 0)
            ),
            0,
        )

        auto_to_cache: Set[int] = set()
        last_user_idx = next(
            (
                i
                for i in range(len(anthropic_messages) - 1, -1, -1)
                if anthropic_messages[i]["role"] == MessageRole.USER
            ),
            None,
        )
        if (
            last_user_idx is not None
            and last_user_idx not in always_cache
            and last_user_idx not in never_cache
            and cache_slots > 0
        ):
            auto_to_cache.add(last_user_idx)
            cache_slots -= 1

        for i, _ in cache_candidates:
            if cache_slots == 0:
                break
            if i in always_cache or i in never_cache or i in auto_to_cache:
                continue
            auto_to_cache.add(i)
            cache_slots -= 1

        return always_cache | auto_to_cache

    @staticmethod
    def _apply_message_cache_control(
        anthropic_messages: Sequence[Dict[str, Any]],
        to_cache: Set[int],
    ) -> None:
        for i in to_cache:
            msg = anthropic_messages[i]
            if msg["content"]:
                msg["content"][-1]["cache_control"] = {"type": "ephemeral"}

    @staticmethod
    def _build_system_blocks(
        system_text: str, should_cache_system: bool
    ) -> Optional[List[Dict[str, Any]]]:
        if not system_text:
            return None
        system_block = {"type": "text", "text": system_text}
        if should_cache_system:
            system_block["cache_control"] = {"type": "ephemeral"}
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
        kwargs = {
            "model": self.model,
            "stream": self.stream,
            **self.api_params,
        }
        kwargs.setdefault("max_tokens", self._max_tokens_cap(self.model))
        (
            kwargs["messages"],
            system_text,
            force_cache_system,
            always_cache,
            never_cache,
        ) = self._collect_request_messages(messages)

        should_cache_system = self._should_cache_system(
            system_text, force_cache_system
        )
        to_cache = self._select_default_cache_targets(
            kwargs["messages"],
            always_cache,
            never_cache,
            should_cache_system,
        )
        self._apply_message_cache_control(kwargs["messages"], to_cache)

        system = self._build_system_blocks(system_text, should_cache_system)
        if system is not None:
            kwargs["system"] = system

        try:
            resp = self._anthropic.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise CompletionError(str(e)) from e

        if isinstance(resp, anthropic.Stream):
            return StreamedClaudeResponse(resp, self)

        res = LLMResponse(
            message=self._render_assistant_message(resp.role, resp.content)
        )
        self._set_response_usage(res, resp.model, self._extract_usage(resp))
        return res

    def get_best_model(self):
        return "claude-opus-4-7"

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
        valid_opts = (
            frozenset(
                inspect.signature(
                    self._anthropic.messages.create
                ).parameters.keys()
            )
            - special_opts
        )
        clamped = {"temperature": (0, 1)}

        for opt in params:
            if opt not in valid_opts:
                raise InvalidAPIParameterError(f"Unknown parameter {opt}")
            if opt == "max_tokens":
                hi = self.__class__._max_tokens_cap(self.model)
                params[opt] = self.__class__._clamp(params[opt], 0, hi)
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
