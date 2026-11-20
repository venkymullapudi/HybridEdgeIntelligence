from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.config import Settings
from hybrid_edge.server import build_orchestrator
from hybrid_edge.telemetry import RecordSink


class FakeMessages:
    def __init__(self, parent: "FakeClaude"):
        self.parent = parent

    async def create(self, **kwargs: Any):
        self.parent.calls.append(kwargs)
        if self.parent.error is not None:
            raise self.parent.error
        content = [SimpleNamespace(type="text", text=self.parent.text)]
        return SimpleNamespace(
            content=content,
            stop_reason=self.parent.stop_reason,
            model=kwargs["model"],
            usage=SimpleNamespace(input_tokens=11, output_tokens=22, iterations=None),
        )


class FakeClaude:
    """Stands in for anthropic.AsyncAnthropic: only `.beta.messages.create` is used."""

    def __init__(self, text: str = "Cloud answer.", stop_reason: str = "end_turn", error: Optional[Exception] = None):
        self.text = text
        self.stop_reason = stop_reason
        self.error = error
        self.calls: List[Dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=FakeMessages(self))

    async def close(self) -> None:
        return None


class MemorySink(RecordSink):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.kind = "memory"
        self.records: List[Dict[str, Any]] = []

    def should_record(self, escalated: bool) -> bool:
        return True

    def submit(self, record: Dict[str, Any]) -> None:
        self.records.append(record)


@pytest.fixture
def settings() -> Settings:
    return Settings.from_dict({
        "edge_backend": "mock",
        "cloud_provider": "anthropic",
        "emit_emf": False,
        "gate_threshold": 0.7,
    })


@pytest.fixture
def fake_claude() -> FakeClaude:
    return FakeClaude()


@pytest.fixture
def orch(settings, fake_claude):
    return build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake_claude),
                              records=MemorySink(settings))
