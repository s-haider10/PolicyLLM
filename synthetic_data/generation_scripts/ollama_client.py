#!/usr/bin/env python3
import json
import urllib.error
import urllib.request
from typing import Optional


class OllamaClient:
    def __init__(
        self,
        model: str = "mistral:latest",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.3,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system:
            payload["system"] = system

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to call Ollama at {self.host}. Is `ollama serve` running and model `{self.model}` available?"
            ) from exc

        text = data.get("response", "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text
