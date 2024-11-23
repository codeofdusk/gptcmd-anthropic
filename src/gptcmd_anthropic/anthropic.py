"""
This module contains an implementation of the Gptcmd LLMProvider for
Anthropic's models.
Copyright 2024 Bill Dengler
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import anthropic
import base64
import inspect

from decimal import Decimal
from typing import Any, Dict, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

from gptcmd.llm import (
    CompletionError,
    InvalidAPIParameterError,
    LLMProvider,
    LLMProviderFeature,
    LLMResponse,
)
from gptcmd.message import Image, Message, MessageRole


class AnthropicProvider(LLMProvider):
    DEFAULT_API_PARAMS = {"max_tokens": 4096}
    SUPPORTED_FEATURES = LLMProviderFeature.RESPONSE_STREAMING

    def __init__(self, client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._anthropic = client
        self._stream = True
        self.update_api_params(self.__class__.DEFAULT_API_PARAMS)

    def _render_message(self, msg: Message) -> Dict[str, Any]:
        return {
            "role": msg.role,
            "content": [
                {"type": "text", "text": msg.content},
                *[self.format_attachment(a) for a in msg.attachments],
            ],
        }

    @classmethod
    def from_config(cls, conf: Dict):
        SPECIAL_OPTS = (
            "model",
            "provider",
        )
        model = conf.get("model")
        client_opts = {k: v for k, v in conf.items() if k not in SPECIAL_OPTS}
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
            "claude-3-5-sonnet-20241022": (
                Decimal("3") / Decimal("1000000"),
                Decimal("15") / Decimal("1000000"),
            ),
            "claude-3-5-haiku-20241022": (
                Decimal("1") / Decimal("1000000"),
                Decimal("5") / Decimal("1000000"),
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

    def complete(self, messages):
        kwargs = {
            "model": self.model,
            "stream": self.stream,
            **self.api_params,
        }
        kwargs["messages"] = []
        system_text = ""

        def _collapse(content1, content2):
            if isinstance(content1, str):
                if isinstance(content2, str):
                    return content1 + content2
                elif isinstance(content2, list):
                    return [{"type": "text", "text": content1}] + content2
            elif isinstance(content1, list):
                if isinstance(content2, str):
                    return content1 + [{"type": "text", "text": content2}]
                elif isinstance(content2, list):
                    return content1 + content2
            else:
                raise TypeError("Unexpected content types")

        for m in messages:
            if m.role == MessageRole.SYSTEM:
                if m.attachments:
                    raise CompletionError(
                        "Attachments on system messages aren't supported"
                    )
                if system_text:
                    system_text += "\n\n" + m.content
                else:
                    system_text = m.content
            else:
                rendered_message = self._render_message(m)
                if (
                    kwargs["messages"]
                    and kwargs["messages"][-1]["role"]
                    == rendered_message["role"]
                ):
                    # Claude doesn't support consecutive messages with the
                    # same role.
                    # "collapse" these spans to one message each.
                    kwargs["messages"][-1]["content"] = _collapse(
                        kwargs["messages"][-1]["content"],
                        rendered_message["content"],
                    )
                else:
                    kwargs["messages"].append(rendered_message)

        cache_weights = []
        for i, msg in enumerate(kwargs["messages"]):
            num_blocks = len(msg["content"])
            total_text_length = sum(
                len(block.get("text", ""))
                for block in msg["content"]
                if block.get("type") == "text"
            )
            w = num_blocks * 1000 + total_text_length
            cache_weights.append((i, w))

        cache_candidates = sorted(
            cache_weights, key=lambda x: x[1], reverse=True
        )

        num_to_mark = 2 if system_text else 3
        for i, _ in cache_candidates[:num_to_mark]:
            msg = kwargs["messages"][i]
            if msg["content"]:
                msg["content"][-1]["cache_control"] = {"type": "ephemeral"}

        # Always cache the last user message
        for msg in reversed(kwargs["messages"]):
            if msg["role"] == "user":
                msg["content"][-1]["cache_control"] = {"type": "ephemeral"}
                break

        if system_text:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        try:
            resp = self._anthropic.beta.prompt_caching.messages.create(
                **kwargs
            )
        except anthropic.APIError as e:
            raise CompletionError(str(e)) from e

        if isinstance(resp, anthropic.Stream):
            return StreamedClaudeResponse(resp, self)
        else:
            msg = Message(
                content="".join(
                    block.text
                    for block in resp.content
                    if block.type == "text"
                ),
                role=resp.role,
            )
            return LLMResponse(
                message=msg,
                prompt_tokens=resp.usage.input_tokens
                + resp.usage.cache_creation_input_tokens,
                sampled_tokens=resp.usage.output_tokens,
                cost_in_cents=self.__class__._estimate_cost_in_cents(
                    model=resp.model,
                    prompt_tokens=resp.usage.input_tokens,
                    cache_write_tokens=resp.usage.cache_creation_input_tokens,
                    cache_read_tokens=resp.usage.cache_read_input_tokens,
                    sampled_tokens=resp.usage.output_tokens,
                ),
            )

    def get_best_model(self):
        return "claude-3-5-sonnet-latest"

    @property
    def valid_models(self):
        # Unfortunately, there doesn't seem to be a programmatic way to get
        # this list
        return (
            "claude-3-5-sonnet-latest",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
        )

    @staticmethod
    def _clamp(val, bottom, top):
        return max(min(val, top), bottom)

    def validate_api_params(self, params):
        SPECIAL_OPTS = frozenset(("model", "messages", "stream", "system"))
        valid_opts = (
            frozenset(
                inspect.signature(
                    self._anthropic.beta.prompt_caching.messages.create
                ).parameters.keys()
            )
            - SPECIAL_OPTS
        )
        CLAMPED = {"temperature": (0, 1), "max_tokens": (0, 4096)}

        for opt in params:
            if opt not in valid_opts:
                raise InvalidAPIParameterError(f"Unknown parameter {opt}")
            elif opt in CLAMPED:
                params[opt] = self.__class__._clamp(params[opt], *CLAMPED[opt])
        return params

    def unset_api_param(self, key):
        super().unset_api_param(key)
        if key in self.__class__.DEFAULT_API_PARAMS:
            self.set_api_param(key, self.__class__.DEFAULT_API_PARAMS[key])
        elif key is None:
            self.update_api_params(self.__class__.DEFAULT_API_PARAMS)


class StreamedClaudeResponse(LLMResponse):
    def __init__(self, backing_stream, provider: AnthropicProvider):
        self._stream = backing_stream
        self._provider = provider
        self._model: str = ""
        self._prompt = 0
        self._cache_write = 0
        self._cache_read = 0
        self._sampled = 0

        m = Message(content="", role="")
        super().__init__(m)

    def _update_usage(self, usage_obj):
        self._prompt += getattr(usage_obj, "input_tokens", 0)
        self._cache_write += getattr(
            usage_obj, "cache_creation_input_tokens", 0
        )
        self._cache_read += getattr(usage_obj, "cache_read_input_tokens", 0)
        self._sampled += getattr(usage_obj, "output_tokens", 0)

    def __iter__(self):
        try:
            for chunk in self._stream:
                if hasattr(chunk, "usage"):
                    self._update_usage(chunk.usage)
                    # This is a final usage chunk
                    # Since we likely haven't been disconnected, update the
                    # real prompt/sampled fields as these results are
                    # likely accurate.
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
        except anthropic.APIError as e:
            raise CompletionError(str(e)) from e


@AnthropicProvider.register_attachment_formatter(Image)
def format_image_for_claude(img):
    try:
        resp = urlopen(img.url)
        mimetype = resp.headers.get("content-type")
        b64data = base64.b64encode(resp.read())
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mimetype,
                "data": b64data.decode("utf-8"),
            },
        }
    except URLError as e:
        raise CompletionError(str(e.reason)) from e
