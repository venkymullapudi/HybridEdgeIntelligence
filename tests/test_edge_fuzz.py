"""Design review question 63: malformed edge responses escalate; they never crash or pass."""

import json
import math
import random

import httpx
import pytest

from hybrid_edge.backends.base import ChatMessage, EdgeBackendError
from hybrid_edge.backends.openai_compat import MalformedResponse, OpenAICompatBackend, parse_chat_completion
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.gate.confidence import Calibrator, ConfidenceGate
from hybrid_edge.gate.signals import Signals
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink

KEYS = ["choices", "message", "content", "logprobs", "token", "logprob", "top_logprobs", "usage",
        "finish_reason", "delta", "prompt_tokens", "completion_tokens"]


def make_junk(r):
    def junk(d=0):
        kinds = ["str", "int", "null", "list", "dict", "float"] if d < 3 else ["str", "int", "null"]
        k = r.choice(kinds)
        return {"str": lambda: r.choice(["", "x", "[DONE]"]), "int": lambda: r.randint(-3, 3), "null": lambda: None,
                "float": lambda: r.choice([1e308, float("nan"), float("inf")]),
                "list": lambda: [junk(d + 1) for _ in range(r.randint(0, 3))],
                "dict": lambda: {r.choice(KEYS): junk(d + 1) for _ in range(r.randint(0, 4))}}[k]()

    def mutate(o):
        if isinstance(o, dict) and o and r.random() < .5:
            k = r.choice(list(o)); o[k] = junk() if r.random() < .5 else mutate(o[k])
        elif isinstance(o, list) and o:
            i = r.randrange(len(o)); o[i] = junk() if r.random() < .5 else mutate(o[i])
        return o
    return mutate


def backend_returning(body):
    b = OpenAICompatBackend("http://e/v1", "m")
    b._client = httpx.AsyncClient(base_url="http://e/v1", transport=httpx.MockTransport(lambda req: httpx.Response(200, text=body)))
    return b


@pytest.mark.parametrize("seed", range(3))
async def test_malformed_edge_responses_only_ever_raise_edge_errors(seed):
    r = random.Random(seed)
    mutate = make_junk(r)
    for _ in range(300):
        choice = {"message": {"content": "Paris"}, "finish_reason": "stop",
                  "logprobs": {"content": [{"token": "Paris", "logprob": -0.1, "top_logprobs": [{"token": "Paris", "logprob": -0.1}]}]}}
        full = mutate({"choices": [choice], "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
        chunk = mutate({"choices": [{"delta": {"content": "Pa"}, "logprobs": {"content": [{"token": "Pa", "logprob": -0.2}]}}]})
        for streaming, body in ((False, json.dumps(full, allow_nan=True)),
                                (True, "data: " + json.dumps(chunk, allow_nan=True) + "\n\ndata: [DONE]\n\n")):
            b = backend_returning(body)
            try:
                if streaming:
                    g = await b.generate_stream([ChatMessage("user", "q")], max_tokens=8, temperature=0.2,
                                                should_abort=lambda t: False)
                else:
                    g = (await b.generate([ChatMessage("user", "q")], max_tokens=8, temperature=0.2))[0]
                assert all(math.isfinite(t.logprob) and t.logprob <= 0 for t in g.tokens)
            except EdgeBackendError:
                pass
            finally:
                await b.aclose()


@pytest.mark.parametrize("payload,why", [
    ({"choices": 3}, "choices is not a list"),
    ({"choices": [None]}, "choice is not an object"),
    ({"choices": [{"logprobs": {"content": [{"token": "a", "logprob": "nan"}]}}]}, "non-finite"),
    ({"choices": [{"logprobs": {"content": [{"token": "a", "logprob": float("inf")}]}}]}, "non-finite"),
    ("[]", "response is not an object"),
])
def test_specific_malformations_are_named(payload, why):
    with pytest.raises(MalformedResponse, match=why):
        parse_chat_completion(payload)


def test_tiny_positive_logprobs_are_clamped():
    g = parse_chat_completion({"choices": [{"message": {"content": "a"},
                                            "logprobs": {"content": [{"token": "a", "logprob": 1e-9}]}}]})[0]
    assert g.tokens[0].logprob == 0.0


@pytest.mark.parametrize("gate", [ConfidenceGate(0.7), ConfidenceGate(0.7, calibrator=Calibrator({"mean_prob": 1.0}, 0.0, 0.5))])
def test_nan_signals_never_score_as_confident(gate):
    n = float("nan")
    d = gate.decide(Signals(mean_prob=n, min_window_prob=n, entropy_conf=n, consistency=n))
    assert d.score == 0.0 and d.escalate


async def test_garbage_edge_response_escalates_end_to_end(settings):
    fake = FakeClaude()
    o = build_orchestrator(settings, backend=backend_returning('data: {"choices": 7}\n\n'),
                           escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    r = await o.infer([ChatMessage("user", "capital of France?")])
    assert r.source == "cloud" and any("edge error" in x for x in r.reasons) and len(fake.calls) == 1
