#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@name: utils.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Wednesday July 09 23:05:15 2025
@desc: Test and compare different large language models on various tasks.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import openai
import anthropic
from datetime import datetime
import os


@dataclass
class BenchmarkResult:
    model_name: str
    task_name: str
    response: str
    latency: float
    input_tokens: int
    output_tokens: int
    cost_estimate: float
    timestamp: datetime
    quality_score: float  # Percentage score (0-100)
    error: Optional[str] = None


@dataclass
class BenchmarkTask:
    name: str
    prompt: str
    category: str
    expected_length: str  # short, medium, long
    evaluation_criteria: List[str]


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
    def __init__(self, api_key: str, model: str = "gpt-4o"):
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
        # GPT-4 pricing (approximate)
        input_cost = input_tokens * 0.0025 / 1000  # $2.50 per 1M tokens
        output_cost = output_tokens * 0.01 / 1000  # $10.00 per 1M tokens
        return input_cost + output_cost


class AnthropicInterface(LLMInterface):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
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
        # Claude 4 Sonnet pricing (approximate)
        input_cost = input_tokens * 0.003 / 1000  # $3.00 per 1M tokens
        output_cost = output_tokens * 0.015 / 1000  # $15.00 per 1M tokens
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
        model: str = "grok-beta",
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
        # Grok pricing
        input_cost = input_tokens * 0.005 / 1000  # $5.00 per 1M tokens
        output_cost = output_tokens * 0.015 / 1000  # $15.00 per 1M tokens
        return input_cost + output_cost


class QualityEvaluator:
    """Evaluates response quality based on task criteria"""

    @staticmethod
    def evaluate_response(task: BenchmarkTask, response: str) -> float:
        """Evaluate response quality and return percentage score (0-100)"""
        if not response or response.strip() == "":
            return 0.0

        score = 0.0

        # Base score for having a response
        score += 20.0

        # Length appropriateness (0-20 points)
        length_score = QualityEvaluator._evaluate_length(task.expected_length, response)
        score += length_score

        # Task-specific evaluation (0-60 points)
        task_score = QualityEvaluator._evaluate_by_task_type(task, response)
        score += task_score

        return min(score, 100.0)

    @staticmethod
    def _evaluate_length(expected_length: str, response: str) -> float:
        """Evaluate if response length matches expectations (0-20 points)"""
        word_count = len(response.split())

        length_targets = {
            "short": (20, 100),  # 20-100 words
            "medium": (100, 400),  # 100-400 words
            "long": (300, 800),  # 300-800 words
        }

        if expected_length not in length_targets:
            return 10.0  # Default score

        min_words, max_words = length_targets[expected_length]

        if min_words <= word_count <= max_words:
            return 20.0
        elif word_count < min_words:
            # Penalise for being too short
            ratio = word_count / min_words
            return 20.0 * ratio
        else:
            # Moderate penalty for being too long
            excess_ratio = (word_count - max_words) / max_words
            penalty = min(excess_ratio * 5, 10)  # Max 10 point penalty
            return max(20.0 - penalty, 5.0)

    @staticmethod
    def _evaluate_by_task_type(task: BenchmarkTask, response: str) -> float:
        """Evaluate based on task category and criteria (0-60 points)"""
        category = task.category.lower()

        if category == "programming":
            return QualityEvaluator._evaluate_code(response)
        elif category == "mathematics":
            return QualityEvaluator._evaluate_math(response)
        elif category == "creative":
            return QualityEvaluator._evaluate_creative(response)
        elif category == "analysis":
            return QualityEvaluator._evaluate_analysis(response)
        elif category == "logic":
            return QualityEvaluator._evaluate_logic(response)
        elif category == "communication":
            return QualityEvaluator._evaluate_communication(response)
        else:
            return QualityEvaluator._evaluate_general(response)

    @staticmethod
    def _evaluate_code(response: str) -> float:
        """Evaluate programming responses (0-60 points)"""
        score = 0.0

        # Check for code blocks or function definitions
        if "def " in response or "function" in response or "```" in response:
            score += 15.0

        # Check for error handling
        if any(
            term in response.lower()
            for term in ["try", "except", "error", "validation"]
        ):
            score += 10.0

        # Check for documentation
        if any(term in response for term in ['"""', "'''", "//", "#"]):
            score += 10.0

        # Check for proper structure
        if "return" in response and (
            "if" in response or "for" in response or "while" in response
        ):
            score += 15.0

        # Check for examples or test cases
        if any(term in response.lower() for term in ["example", "test", "usage"]):
            score += 10.0

        return score

    @staticmethod
    def _evaluate_math(response: str) -> float:
        """Evaluate mathematical reasoning (0-60 points)"""
        score = 0.0

        # Check for step-by-step approach
        if any(
            term in response.lower()
            for term in ["step", "first", "then", "next", "finally"]
        ):
            score += 20.0

        # Check for calculations
        if any(char in response for char in "=+-*/") and any(
            char.isdigit() for char in response
        ):
            score += 15.0

        # Check for units or measurements
        if any(
            term in response.lower()
            for term in ["miles", "hours", "mph", "speed", "time"]
        ):
            score += 15.0

        # Check for clear conclusion
        if any(
            term in response.lower()
            for term in ["therefore", "total", "answer", "result"]
        ):
            score += 10.0

        return score

    @staticmethod
    def _evaluate_creative(response: str) -> float:
        """Evaluate creative writing (0-60 points)"""
        score = 0.0

        # Check for narrative elements
        narrative_indicators = [
            "said",
            "felt",
            "looked",
            "thought",
            "realised",
            "discovered",
        ]
        if any(term in response.lower() for term in narrative_indicators):
            score += 20.0

        # Check for emotional content
        emotion_words = [
            "happy",
            "sad",
            "surprised",
            "confused",
            "excited",
            "worried",
            "wonder",
        ]
        if any(term in response.lower() for term in emotion_words):
            score += 15.0

        # Check for descriptive language
        if len([word for word in response.split() if len(word) > 6]) > 10:
            score += 15.0

        # Check for dialogue or direct speech
        if '"' in response or "'" in response:
            score += 10.0

        return score

    @staticmethod
    def _evaluate_analysis(response: str) -> float:
        """Evaluate analytical responses (0-60 points)"""
        score = 0.0

        # Check for key concepts mentioned
        key_terms = ["correlation", "causation", "relationship", "data", "statistics"]
        if any(term in response.lower() for term in key_terms):
            score += 20.0

        # Check for examples
        if any(
            term in response.lower()
            for term in ["example", "instance", "case", "such as"]
        ):
            score += 15.0

        # Check for practical applications
        if any(
            term in response.lower()
            for term in ["practical", "real-world", "application", "important"]
        ):
            score += 15.0

        # Check for clear structure
        if any(
            term in response.lower()
            for term in ["first", "second", "however", "furthermore"]
        ):
            score += 10.0

        return score

    @staticmethod
    def _evaluate_logic(response: str) -> float:
        """Evaluate logical reasoning (0-60 points)"""
        score = 0.0

        # Check for logical structure
        if any(
            term in response.lower()
            for term in ["premise", "conclusion", "therefore", "because"]
        ):
            score += 20.0

        # Check for reasoning process
        if any(
            term in response.lower()
            for term in ["valid", "invalid", "logical", "reasoning"]
        ):
            score += 15.0

        # Check for clear explanation
        if len(response.split()) > 50:  # Adequate length for explanation
            score += 15.0

        # Check for structured argument
        if any(
            term in response.lower() for term in ["cannot", "can", "conclude", "assume"]
        ):
            score += 10.0

        return score

    @staticmethod
    def _evaluate_communication(response: str) -> float:
        """Evaluate communication advice (0-60 points)"""
        score = 0.0

        # Check for key communication concepts
        comm_terms = [
            "listen",
            "clear",
            "communicate",
            "understand",
            "feedback",
            "respect",
        ]
        if any(term in response.lower() for term in comm_terms):
            score += 20.0

        # Check for actionable advice
        if any(
            term in response.lower()
            for term in ["should", "can", "try", "practice", "avoid"]
        ):
            score += 15.0

        # Check for professional context
        if any(
            term in response.lower()
            for term in ["professional", "workplace", "colleague", "meeting"]
        ):
            score += 15.0

        # Check for organisation
        if any(term in response for term in ["1.", "2.", "•", "-", "first", "second"]):
            score += 10.0

        return score

    @staticmethod
    def _evaluate_general(response: str) -> float:
        """General evaluation for uncategorised tasks (0-60 points)"""
        score = 0.0

        # Basic completeness
        word_count = len(response.split())
        if word_count > 20:
            score += 20.0
        elif word_count > 10:
            score += 10.0

        # Check for structured thinking
        if any(
            term in response.lower()
            for term in ["because", "therefore", "however", "although"]
        ):
            score += 15.0

        # Check for examples or specifics
        if any(
            term in response.lower()
            for term in ["example", "such as", "specifically", "instance"]
        ):
            score += 15.0

        # Basic coherence check (sentences and punctuation)
        sentence_count = response.count(".") + response.count("!") + response.count("?")
        if sentence_count >= 3:
            score += 10.0

        return score


class APIKeyManager:
    """Secure API key management"""

    @staticmethod
    def load_from_env() -> Dict[str, Optional[str]]:
        """Load API keys from environment variables"""
        return {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_TOKEN"),
            "xai": os.getenv("XAI_API_KEY") or os.getenv("XAI_TOKEN"),
        }

    @staticmethod
    def load_from_file(config_file: str = ".env") -> Dict[str, Optional[str]]:
        """Load API keys from configuration file"""
        keys = {}
        try:
            with open(config_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        key_name = key.strip().lower()
                        # Normalise key names
                        if "openai" in key_name:
                            key_name = "openai"
                        elif "anthropic" in key_name:
                            key_name = "anthropic"
                        elif "deepseek" in key_name:
                            key_name = "deepseek"
                        elif "xai" in key_name:
                            key_name = "xai"

                        keys[key_name] = value.strip().strip('"').strip("'")
        except FileNotFoundError:
            print(f"Configuration file {config_file} not found")
        return keys


def setup_environment():
    """Helper function to guide users through API key setup"""
    print("=== API Key Setup Guide ===\n")

    print("You need API keys from these providers:")
    print("1. OpenAI: https://platform.openai.com/api-keys")
    print("2. Anthropic: https://console.anthropic.com/")
    print("3. DeepSeek: https://platform.deepseek.com/")
    print("4. xAI: https://x.ai/api")

    print("\n=== Recommended Setup Method ===")
    print("Create a .env file in your project directory:")
    print()
    print("# .env file content")
    print("OPENAI_API_KEY=sk-proj-...")
    print("ANTHROPIC_API_KEY=sk-ant-...")
    print("DEEPSEEK_API_KEY=sk-...")
    print("XAI_API_KEY=xai-...")
    print()
    print("Then run: python benchmark.py")

    print("\n=== Alternative: Environment Variables ===")
    print("Linux/Mac:")
    print("export OPENAI_API_KEY='sk-proj-...'")
    print("export ANTHROPIC_API_KEY='sk-ant-...'")
    print("export DEEPSEEK_API_KEY='sk-...'")
    print("export XAI_API_KEY='xai-...'")
    print()
    print("Windows:")
    print("set OPENAI_API_KEY=sk-proj-...")
    print("set ANTHROPIC_API_KEY=sk-ant-...")
    print("set DEEPSEEK_API_KEY=sk-...")
    print("set XAI_API_KEY=xai-...")
