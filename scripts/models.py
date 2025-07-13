#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@name: models.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Saturday July 12 13:02:25 2025
@updated: Sunday July 13 15:30:00 2025
@desc: LLM model interfaces for benchmarking
"""

import time
from typing import Dict, Tuple, Any
from abc import ABC, abstractmethod
import openai
import anthropic
import google.generativeai as genai


class LLMInterface(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    async def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Generate response and return (response, metadata)"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pass


class OpenAIInterface(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
            )

            latency = time.time() - start_time

            metadata = {
                "latency": latency,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "error": None,
            }

            return response.choices[0].message.content, metadata

        except Exception as e:
            return "", {
                "latency": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }

    def get_model_name(self) -> str:
        return f"OpenAI-{self.model}"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # GPT-4.1 pricing (approximate)
        input_cost = input_tokens * 0.005 / 1000  # $5.00 per 1M tokens
        output_cost = output_tokens * 0.02 / 1000  # $20.00 per 1M tokens
        return input_cost + output_cost


class AnthropicInterface(LLMInterface):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            latency = time.time() - start_time

            metadata = {
                "latency": latency,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "error": None,
            }

            return response.content[0].text, metadata

        except Exception as e:
            return "", {
                "latency": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }

    def get_model_name(self) -> str:
        return f"Anthropic-{self.model}"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Claude Sonnet 4 pricing (approximate)
        input_cost = input_tokens * 0.005 / 1000  # $5.00 per 1M tokens
        output_cost = output_tokens * 0.025 / 1000  # $25.00 per 1M tokens
        return input_cost + output_cost


class DeepSeekInterface(LLMInterface):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
            )

            latency = time.time() - start_time

            metadata = {
                "latency": latency,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "error": None,
            }

            return response.choices[0].message.content, metadata

        except Exception as e:
            return "", {
                "latency": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }

    def get_model_name(self) -> str:
        return f"DeepSeek-{self.model}"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # DeepSeek V3 pricing (very competitive)
        input_cost = input_tokens * 0.00027 / 1000  # $0.27 per 1M tokens
        output_cost = output_tokens * 0.0011 / 1000  # $1.10 per 1M tokens
        return input_cost + output_cost


class XAIInterface(LLMInterface):
    """Interface for Grok models via xAI API"""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-4-0709",
        base_url: str = "https://api.x.ai/v1",
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
            )

            latency = time.time() - start_time

            metadata = {
                "latency": latency,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "error": None,
            }

            return response.choices[0].message.content, metadata

        except Exception as e:
            return "", {
                "latency": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }

    def get_model_name(self) -> str:
        return f"xAI-{self.model}"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Grok 4 pricing (approximate)
        input_cost = input_tokens * 0.01 / 1000  # $10.00 per 1M tokens
        output_cost = output_tokens * 0.03 / 1000  # $30.00 per 1M tokens
        return input_cost + output_cost


class GeminiInterface(LLMInterface):
    """Interface for Google Gemini models"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    async def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=2000,
                temperature=0.7,
            )

            response = await self.model.generate_content_async(
                prompt, generation_config=generation_config
            )

            latency = time.time() - start_time

            # Extract token usage if available
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0

            metadata = {
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            }

            return response.text, metadata

        except Exception as e:
            return "", {
                "latency": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }

    def get_model_name(self) -> str:
        return f"Google-{self.model_name}"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Gemini 2.5 Pro pricing (approximate)
        input_cost = input_tokens * 0.001 / 1000  # $1.00 per 1M tokens
        output_cost = output_tokens * 0.004 / 1000  # $4.00 per 1M tokens
        return input_cost + output_cost


class MoonshotInterface(LLMInterface):
    """Interface for Moonshot AI models"""

    def __init__(
        self,
        api_key: str,
        model: str = "moonshot-v1-8k",
        base_url: str = "https://api.moonshot.ai/v1",
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
            )

            latency = time.time() - start_time

            metadata = {
                "latency": latency,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "error": None,
            }

            return response.choices[0].message.content, metadata

        except Exception as e:
            return "", {
                "latency": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }

    def get_model_name(self) -> str:
        return f"Moonshot-{self.model}"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Moonshot pricing (competitive rates)
        input_cost = input_tokens * 0.001 / 1000  # $1.00 per 1M tokens
        output_cost = output_tokens * 0.002 / 1000  # $2.00 per 1M tokens
        return input_cost + output_cost


class QwenInterface(LLMInterface):
    """Interface for Qwen3 32B via OpenRouter"""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3-32b:free",
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "https://llmbenchmark.local",
        site_name: str = "LLM Benchmark Suite",
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.site_url = site_url
        self.site_name = site_name

    async def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                extra_body={},
            )

            latency = time.time() - start_time

            metadata = {
                "latency": latency,
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "error": None,
            }

            return response.choices[0].message.content, metadata

        except Exception as e:
            return "", {
                "latency": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }

    def get_model_name(self) -> str:
        return f"Qwen-{self.model.replace('qwen/', '').replace(':free', '')}"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Qwen3 32B is free via OpenRouter
        return 0.0
