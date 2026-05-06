# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.logger import init_logger

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

logger = init_logger(__name__)


class TerratorchRenderer(BaseRenderer):
    @classmethod
    def from_config(
        cls,
        config: VllmConfig,  # type: ignore[override]
        tokenizer_kwargs: dict[str, Any],
    ) -> "TerratorchRenderer":
        model_config = config.model_config
        if not model_config.skip_tokenizer_init:
            raise ValueError("Terratorch renderer requires `skip_tokenizer_init=True`")

        return cls(config, None)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        model_config = self.model_config

        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        if mm_data is not None:
            _media_dl = mm_data.pop("__media_download_time__", 0.0)
        else:
            _media_dl = 0.0

        prompt = parse_dec_only_prompt([1])  # Dummy token IDs
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        # Thread media download time from mm_data into the prompt
        # dict so it survives asyncio.gather() context isolation.
        if _media_dl > 0:
            prompt["media_download_time"] = _media_dl

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        model_config = self.model_config

        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        if mm_data is not None:
            _media_dl = mm_data.pop("__media_download_time__", 0.0)
        else:
            _media_dl = 0.0

        prompt = parse_dec_only_prompt([1])  # Dummy token IDs
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        # Thread media download time from mm_data into the prompt
        # dict so it survives asyncio.gather() context isolation.
        if _media_dl > 0:
            prompt["media_download_time"] = _media_dl

        return conversation, prompt
