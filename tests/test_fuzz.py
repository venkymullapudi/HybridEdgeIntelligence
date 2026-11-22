"""Design review question 62: no client input may cause a 500.

Seeded random fuzzing of both inference endpoints, fully offline (mock edge, fake
Claude client). Any 5xx, or a streamed `error` event with status 500, fails the test.
"""

import json
import random

import pytest
from fastapi.testclient import TestClient

from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink

ATOMS = ["", " ", "\x00", "‮", "\ud800", "\udfff", "😀", "中文", "\n\r\t", "<script>", "'''",
         "A" * 3000, "\\", '"', "%s%n", "${x}", "\u0085", "﻿", "\U0010ffff",
         "hard", "easy", "essay", "unsure"]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def text(self):
        return "".join(self.r.choice(ATOMS) for _ in range(self.r.randint(0, 5)))

    def val(self, depth=0):
        kinds = ["str", "int", "float", "bool", "null", "big", "nan", "inf", "list", "dict"]
        k = self.r.choice(kinds if depth < 2 else kinds[:8])
        r = self.r
        return {
            "str": self.text, "int": lambda: r.randint(-5, 10 ** 6), "float": lambda: r.uniform(-2, 2),
            "bool": lambda: r.random() < .5, "null": lambda: None, "big": lambda: 10 ** 40,
            "nan": lambda: float("nan"), "inf": lambda: float("-inf"),
            "list": lambda: [self.val(depth + 1) for _ in range(r.randint(0, 3))],
            "dict": lambda: {self.text()[:8]: self.val(depth + 1) for _ in range(r.randint(0, 3))},
        }[k]()

    def body(self):
        r, b = self.r, {}
        if r.random() < .6:
            b["messages"] = [{"role": r.choice(["user", "assistant", "system", "tool", "", None, 5]),
                              "content": self.val()} for _ in range(r.randint(0, 5))] if r.random() < .9 else self.val()
        if r.random() < .5:
            b["prompt"] = self.val() if r.random() < .3 else self.text()
        for f in ("route", "min_confidence", "max_tokens", "use_cache", "include_signals", "extra"):
            if r.random() < .3:
                b[f] = self.val() if r.random() < .5 else r.choice(["auto", "edge", "cloud", 0.5, 64, True])
        raw = json.dumps(b, allow_nan=True)
        if r.random() < .1:
            raw = raw[: r.randint(0, 40)]  # truncated / malformed JSON
        return raw.encode("utf-8", "surrogatepass")


@pytest.mark.parametrize("seed", range(4))
def test_no_input_causes_a_server_error(settings, seed):
    settings.edge_deadline_s = 2
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    gen = Gen(seed)
    with TestClient(create_app(settings, o), raise_server_exceptions=False) as c:
        for _ in range(150):
            raw = gen.body()
            for path in ("/v1/infer", "/v1/infer/stream"):
                r = c.post(path, content=raw, headers={"content-type": "application/json"})
                assert r.status_code < 500, (path, r.status_code, raw[:200])
                if path.endswith("stream") and r.status_code == 200:
                    assert '"status": 500' not in r.text, (path, raw[:200])


def test_validation_errors_never_echo_input(settings):
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    with TestClient(create_app(settings, o)) as c:
        for raw in ['{"prompt": NaN}', '{"prompt": Infinity, "route": "x"}', '{"prompt": ["secret-token-123"]}']:
            r = c.post("/v1/infer", content=raw, headers={"content-type": "application/json"})
            assert r.status_code == 422, raw
            assert "secret-token-123" not in r.text and "input" not in r.json()["detail"][0]
            assert {"loc", "msg", "type"} <= set(r.json()["detail"][0])
