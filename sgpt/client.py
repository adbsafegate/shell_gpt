import json
from pathlib import Path
from typing import Dict, Generator, List

import openai

from .cache import Cache
from .config import cfg

CACHE_LENGTH = int(cfg.get("CACHE_LENGTH"))
CACHE_PATH = Path(cfg.get("CACHE_PATH"))
REQUEST_TIMEOUT = int(cfg.get("REQUEST_TIMEOUT"))
DISABLE_STREAMING = str(cfg.get("DISABLE_STREAMING"))
openai.api_type = "azure"
openai.api_version = cfg.get("AZURE_API_VERSION")


class OpenAIClient:
    cache = Cache(CACHE_LENGTH, CACHE_PATH)

    def __init__(self, api_host: str, api_key: str) -> None:
        openai.api_base = api_host
        openai.api_key = api_key

    @cache
    def _request(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-35-turbo",
        temperature: float = 1,
        top_probability: float = 1,
    ) -> Generator[str, None, None]:
        """
        Make request to OpenAI API, read more:
        https://platform.openai.com/docs/api-reference/chat

        :param messages: List of messages {"role": user or assistant, "content": message_string}
        :param model: String gpt-3.5-turbo or gpt-3.5-turbo-0301
        :param temperature: Float in 0.0 - 2.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :return: Response body JSON.
        """
        stream = DISABLE_STREAMING == "false"
        response = openai.ChatCompletion.create(
            engine=model,
            messages=messages,
            temperature=temperature,
            top_p=top_probability,
            stream=True

        )

        if not stream:
            data = response
            yield data["choices"][0]["message"]["content"]  # type: ignore
            return
        for line in response:
            data = line
            if data == "[DONE]":  # type: ignore
                break
            if not data:
                continue
            delta = data["choices"][0]["delta"]  # type: ignore
            if "content" not in delta:
                continue
            yield delta["content"]

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 1,
        top_probability: float = 1,
        caching: bool = True,
    ) -> Generator[str, None, None]:
        """
        Generates single completion for prompt (message).

        :param messages: List of dict with messages and roles.
        :param model: String gpt-3.5-turbo or gpt-3.5-turbo-0301.
        :param temperature: Float in 0.0 - 1.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :param caching: Boolean value to enable/disable caching.
        :return: String generated completion.
        """
        yield from self._request(
            messages,
            model,
            temperature,
            top_probability,
            caching=caching,
        )
