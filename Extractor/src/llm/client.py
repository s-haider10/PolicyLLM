"""LLM client wrapper for local (Ollama) and cloud providers with JSON schema enforcement."""
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Type

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except ImportError:  # boto3 not required for stub/local providers
    boto3 = None

    class _BotoStub(Exception):
        pass

    BotoCoreError = ClientError = NoCredentialsError = _BotoStub
from pydantic import BaseModel, ValidationError


_OPENAI_PRICE_PER_1M = {
    "chat": {
        "gpt-4o-mini": (0.15, 0.60),  # input, output USD per 1M tokens
        "gpt-4o": (2.50, 10.00),
    },
    "embedding": {
        "text-embedding-3-small": (0.02, 0.0),
        "text-embedding-3-large": (0.13, 0.0),
    },
}


def _resolve_openai_price(kind: str, model_id: str) -> tuple[float, float] | None:
    table = _OPENAI_PRICE_PER_1M.get(kind, {})
    for prefix in sorted(table.keys(), key=len, reverse=True):
        if model_id.startswith(prefix):
            return table[prefix]
    return None


def _budget_state_path() -> Path:
    return Path(
        os.getenv("POLICYLLM_BUDGET_STATE_PATH")
        or os.getenv("POLICYLLM_COST_STATE_PATH")
        or "results/api_budget_usage.json"
    )


def _openai_timeout_seconds() -> float:
    raw = os.getenv("POLICYLLM_OPENAI_TIMEOUT_SEC", "120")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 120.0
    return max(1.0, value)


def record_openai_usage(kind: str, model_id: str, input_tokens: int, output_tokens: int = 0) -> None:
    """Record estimated OpenAI spend and enforce optional hard budget cap.

    Enabled when POLICYLLM_BUDGET_USD is set.
    """
    if input_tokens <= 0 and output_tokens <= 0:
        return

    budget_raw = os.getenv("POLICYLLM_BUDGET_USD")
    if not budget_raw:
        return
    try:
        budget_usd = float(budget_raw)
    except ValueError:
        return
    if budget_usd <= 0:
        return

    price = _resolve_openai_price(kind=kind, model_id=model_id)
    if price is None:
        return
    in_per_1m, out_per_1m = price
    est = (max(0, input_tokens) / 1_000_000.0) * in_per_1m + (max(0, output_tokens) / 1_000_000.0) * out_per_1m
    if est <= 0:
        return

    path = _budget_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "budget_usd": budget_usd,
        "estimated_spend_usd": 0.0,
        "by_model": {},
        "updated_at_utc": None,
    }
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                state.update(loaded)
        except Exception:
            pass

    spent = float(state.get("estimated_spend_usd", 0.0))
    projected = spent + est
    if projected > budget_usd:
        raise RuntimeError(
            f"Budget guard triggered: projected spend ${projected:.4f} exceeds POLICYLLM_BUDGET_USD=${budget_usd:.2f} "
            f"for {kind} model '{model_id}'."
        )

    by_model = state.setdefault("by_model", {})
    model_state = by_model.setdefault(
        model_id,
        {
            "kind": kind,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_spend_usd": 0.0,
        },
    )
    model_state["input_tokens"] = int(model_state.get("input_tokens", 0)) + max(0, input_tokens)
    model_state["output_tokens"] = int(model_state.get("output_tokens", 0)) + max(0, output_tokens)
    model_state["estimated_spend_usd"] = float(model_state.get("estimated_spend_usd", 0.0)) + est

    state["budget_usd"] = budget_usd
    state["estimated_spend_usd"] = projected
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


class LLMClient:
    """Provide a unified interface over local Ollama and cloud APIs."""

    def __init__(
        self,
        provider: str,
        model_id: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        region: str = "us-east-2",
        top_k: Optional[int] = None,
        retries: int = 2,
        backoff: float = 1.5,
    ):
        self.provider = provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.region = region
        self.top_k = top_k
        self.retries = retries
        self.backoff = backoff
        self._bedrock = (
            boto3.client("bedrock-runtime", region_name=region) if provider == "bedrock_claude" else None
        )
        self._openai = None
        self._anthropic = None
        self._stub = provider == "stub"

        if provider in ("chatgpt", "ollama"):
            try:
                from openai import OpenAI

                if provider == "ollama":
                    self._openai = OpenAI(
                        base_url="http://localhost:11434/v1",
                        api_key="ollama",
                        timeout=_openai_timeout_seconds(),
                    )
                else:
                    self._openai = OpenAI(timeout=_openai_timeout_seconds())
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("OpenAI client not available; install openai>=1.10.0") from exc
        elif provider == "anthropic":
            try:
                import anthropic

                self._anthropic = anthropic.Anthropic()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("Anthropic client not available; install anthropic>=0.18.1") from exc

    def invoke_json(self, prompt: str, schema: Optional[Type[BaseModel] | Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call the selected provider and return parsed JSON matching schema."""
        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                if self.provider == "bedrock_claude":
                    raw_text = self._invoke_bedrock(prompt)
                elif self.provider in ("chatgpt", "ollama"):
                    raw_text = self._invoke_openai(prompt)
                elif self.provider == "anthropic":
                    raw_text = self._invoke_anthropic(prompt)
                elif self._stub:
                    raw_text = self._invoke_stub(prompt)
                else:
                    raise NotImplementedError(f"Provider {self.provider} not implemented")
                parsed = self._coerce_json(raw_text)
                return self._validate(parsed, schema)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.backoff ** attempt)
        raise RuntimeError(f"LLM invocation failed after retries: {last_err}") from last_err

    def _invoke_bedrock(self, prompt: str) -> str:
        """Invoke Claude Sonnet via AWS Bedrock Converse API and return text."""
        try:
            kwargs = {
                "modelId": self.model_id,
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stopSequences": [],
                },
                "performanceConfig": {"latency": "standard"},
            }
            if self.top_k is not None:
                kwargs["additionalModelRequestFields"] = {"top_k": self.top_k}

            try:
                resp = self._bedrock.converse(**kwargs)
            except NoCredentialsError as exc:
                raise RuntimeError(
                    "AWS credentials not found. Configure env vars, shared credentials, or an IAM role."
                ) from exc
            content = resp.get("output", {}).get("message", {}).get("content", [])
            if not content:
                raise ValueError("Empty response content from Bedrock")
            return content[0]["text"]
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(f"Bedrock invocation error: {exc}") from exc

    def _invoke_openai(self, prompt: str) -> str:
        """Invoke LLM via OpenAI-compatible API (ChatGPT or Ollama)."""
        if not self._openai:
            raise RuntimeError("OpenAI client not initialized")
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.provider == "chatgpt":
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._openai.chat.completions.create(**kwargs)
        if self.provider == "chatgpt":
            usage = getattr(resp, "usage", None)
            if usage is not None:
                record_openai_usage(
                    kind="chat",
                    model_id=self.model_id,
                    input_tokens=_usage_value(usage, "prompt_tokens"),
                    output_tokens=_usage_value(usage, "completion_tokens"),
                )
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from OpenAI")
        return content

    def _invoke_anthropic(self, prompt: str) -> str:
        """Invoke Claude via Anthropic API and return text."""
        if not self._anthropic:
            raise RuntimeError("Anthropic client not initialized")
        resp = self._anthropic.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.content[0].text if resp.content else ""
        if not content:
            raise ValueError("Empty response content from Anthropic")
        return content

    def _invoke_stub(self, prompt: str) -> str:
        """Stub responses for offline/testing."""
        low = prompt.lower()
        if "extract structured fields" in low:
            # Simple heuristic stub for the sample return policy text
            if "return items within 30 days" in low and "store credit" in low:
                return json.dumps(
                    {
                        "scope": {
                            "customer_segments": ["all"],
                            "product_categories": ["all"],
                            "channels": ["all"],
                            "regions": ["all"],
                        },
                        "conditions": [
                            {
                                "type": "time_window",
                                "value": 30,
                                "unit": "days",
                                "operator": "<=",
                                "target": "general",
                                "source_text": "Customers may return items within 30 days of purchase for a full refund with receipt.",
                            },
                            {
                                "type": "time_window",
                                "value": 15,
                                "unit": "days",
                                "operator": "<=",
                                "target": "electronics",
                                "source_text": "Electronics must be returned within 15 days.",
                            },
                            {
                                "type": "boolean_flag",
                                "value": True,
                                "parameter": "has_receipt",
                                "source_text": "full refund with receipt",
                            },
                        ],
                        "actions": [
                            {
                                "type": "required",
                                "action": "full_refund",
                                "requires": ["has_receipt", "within_window"],
                                "source_text": "Customers may return items within 30 days of purchase for a full refund with receipt.",
                            },
                            {
                                "type": "fallback",
                                "action": "store_credit",
                                "requires": ["no_receipt"],
                                "source_text": "Items without a receipt receive store credit only.",
                            },
                        ],
                        "exceptions": [],
                    }
                )
            return json.dumps(
                {
                    "scope": {
                        "customer_segments": ["all"],
                        "product_categories": ["all"],
                        "channels": ["all"],
                        "regions": ["all"],
                    },
                    "conditions": [],
                    "actions": [],
                    "exceptions": [],
                }
            )
        if "classify section" in low or "policy extraction assistant" in low:
            return json.dumps({"is_policy": True, "confidence": 0.95, "reason": "stubbed policy-like content"})
        if "metadata annotator" in low:
            return json.dumps(
                {"owner": "unknown", "effective_date": None, "domain": "refund", "regulatory_linkage": []}
            )
        if "validation assistant" in low:
            return json.dumps({"issues": [], "needs_review": False, "confidence": 0.9})
        return json.dumps({})

    @staticmethod
    def _coerce_json(raw_text: str) -> Dict[str, Any] | Any:
        """Best-effort JSON parsing: try direct loads, then extract first JSON object/array."""
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        # Attempt to extract first JSON object or array
        match = re.search(r"(\{.*\}|\[.*\])", raw_text, flags=re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # Enhanced error logging for debugging
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"JSON decode error at position {e.pos}: {e.msg}")
                logger.error(f"Full JSON (first 1000 chars):\n{json_str[:1000]}")
                raise ValueError(f"Malformed JSON at char {e.pos}: {e.msg}. Check logs for full JSON.") from e
        raise ValueError(f"Unable to parse JSON from response: {raw_text[:200]}")

    @staticmethod
    def _validate(payload: Dict[str, Any], schema: Optional[Type[BaseModel] | Dict[str, Any]]) -> Dict[str, Any]:
        if schema is None:
            return payload
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                return schema.model_validate(payload).model_dump()
            except ValidationError as exc:
                raise ValueError(f"Payload validation failed: {exc}") from exc
        # If schema is a dict (placeholder), return as-is.
        return payload
