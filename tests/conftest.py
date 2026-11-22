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

    def stream(self, **kwargs: Any) -> FakeStream:
        return FakeStream(self, kwargs)

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


class FakeStream:
    def __init__(self, messages: "FakeMessages", kwargs):
        self.m, self.kwargs = messages, kwargs

    async def __aenter__(self):
        self.final = await self.m.create(**self.kwargs)  # records the call, raises the error
        text = self.m.parent.text

        async def gen():
            for i in range(0, len(text), self.m.parent.chunk):
                if self.m.parent.stream_error_after is not None and i >= self.m.parent.stream_error_after:
                    raise self.m.parent.stream_error
                yield text[i:i + self.m.parent.chunk]

        self.text_stream = gen()
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.final


class FakeClaude:
    """Stands in for anthropic.AsyncAnthropic: only `.beta.messages.create` is used."""

    def __init__(self, text: str = "Cloud answer.", stop_reason: str = "end_turn", error: Optional[Exception] = None):
        self.text = text
        self.stop_reason = stop_reason
        self.error = error
        self.calls: List[Dict[str, Any]] = []
        self.chunk = 5
        self.stream_error_after: Optional[int] = None
        self.stream_error: Optional[Exception] = None
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


@pytest.fixture(autouse=True)
def _no_real_claude(monkeypatch):
    """Tests must never reach the real Anthropic API: the SDK finds credentials on its
    own (env, `ant auth login` profiles), so an un-faked client would make billed calls."""
    import anthropic

    def refuse(*a, **k):
        raise AssertionError("test tried to build a real Anthropic client; pass a fake client")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", refuse)
    monkeypatch.setattr(anthropic, "AsyncAnthropicBedrockMantle", refuse)
