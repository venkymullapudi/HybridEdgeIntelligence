import json
import random

import pytest

from hybrid_edge.backends.openai_compat import parse_chat_completion
from hybrid_edge.calibrate import choose_threshold, fit, main as calibrate_main
from hybrid_edge.cache import TTLCache, cache_key
from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cloud.claude import CircuitBreaker, TokenBucket
from hybrid_edge.config import Settings


def test_parse_vllm_response():
    payload = {
        "choices": [{
            "message": {"content": "Paris"},
            "finish_reason": "stop",
            "logprobs": {"content": [{"token": "Paris", "logprob": -0.01,
                                      "top_logprobs": [{"token": "Paris", "logprob": -0.01},
                                                       {"token": "Lyon", "logprob": -5.0}]}]},
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1},
    }
    g = parse_chat_completion(payload, 3.0)[0]
    assert g.text == "Paris" and g.tokens[0].top["Lyon"] == -5.0 and g.prompt_tokens == 12


def test_settings_env(monkeypatch):
    monkeypatch.setenv("HEI_GATE_THRESHOLD", "0.55")
    monkeypatch.setenv("HEI_API_KEYS", "a, b")
    monkeypatch.setenv("HEI_EMIT_EMF", "false")
    monkeypatch.setenv("HEI_GATE_WEIGHTS", "{mean_prob: 1, min_window_prob: 0, entropy: 0, consistency: 0}")
    s = Settings.load()
    assert s.gate_threshold == 0.55 and s.api_keys == ["a", "b"] and s.emit_emf is False
    assert s.gate_weights.mean_prob == 1


def test_settings_validation():
    with pytest.raises(ValueError):
        Settings.from_dict({"gate_threshold": 2})
    with pytest.raises(ValueError):
        Settings.from_dict({"nope": 1})
    with pytest.raises(ValueError):
        Settings.from_dict({"record_sink": "firehose"})


def test_cache_ttl_and_lru():
    c = TTLCache(maxsize=2, ttl_s=100)
    c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 1
    assert cache_key([ChatMessage("user", " a b\r\n")]) == cache_key([ChatMessage("user", "a b")])
    # Indentation is meaning (code, YAML): must not collide.
    assert cache_key([ChatMessage("user", "if x:\n    y()\nz()")]) != cache_key([ChatMessage("user", "if x:\n    y()\n    z()")])


def test_token_bucket_and_breaker():
    b = TokenBucket(2)
    assert b.take() and b.take() and not b.take()
    assert TokenBucket(0).take()
    br = CircuitBreaker(failures=1, cooldown_s=1000)
    br.failure()
    assert br.open
    br.success()
    assert not br.open


def _synthetic(n=400, seed=1):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        good = rng.random() < 0.6
        mp = rng.uniform(0.75, 0.99) if good else rng.uniform(0.3, 0.85)
        rows.append(({"mean_prob": mp, "min_window_prob": mp * 0.9, "entropy_conf": mp, "consistency": 1.0,
                      "hedge": 0.0, "truncated": 0.0, "log_len": 3.0}, int(good)))
    return rows


def test_calibration_fit_and_threshold():
    rows = _synthetic()
    cal = fit(rows)
    assert cal.weights["mean_prob"] > 0
    thr, stats = choose_threshold(cal, rows, 0.95)
    assert stats["edge_precision"] >= 0.95 and 0 < stats["edge_coverage"] < 1


def test_calibrate_cli(tmp_path, capsys):
    path = tmp_path / "rec.jsonl"
    with open(path, "w") as fh:
        for feats, y in _synthetic():
            fh.write(json.dumps({"features": feats, "label": y}) + "\n")
    out = tmp_path / "cal.json"
    calibrate_main([str(path), "-o", str(out)])
    cal = json.loads(out.read_text())
    assert 0 < cal["threshold"] < 1


def test_main_sets_graceful_shutdown_above_cloud_deadline(monkeypatch):
    import uvicorn
    from hybrid_edge import server

    seen = {}
    monkeypatch.setenv("HEI_EDGE_BACKEND", "mock")
    monkeypatch.setenv("HEI_CLOUD_PROVIDER", "none")
    monkeypatch.setenv("HEI_CLOUD_DEADLINE_S", "30")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    server.main()
    assert seen["timeout_graceful_shutdown"] == 35


def test_latency_histogram_is_cumulative_prometheus():
    from hybrid_edge.telemetry import LATENCY_BUCKETS_MS, Metrics

    m = Metrics(Settings.from_dict({"emit_emf": False}))
    for ms in (10, 25, 30, 3000, 200000):
        m.observe("cloud", True, False, 0.5, ms)
    text = m.prometheus()

    def bucket(le):
        line = next(x for x in text.splitlines() if 'source="cloud",le="%s"' % le in x)
        return int(line.rsplit(" ", 1)[1])

    assert bucket(25) == 2          # le is inclusive: 10 and 25
    assert bucket(50) == 3
    assert bucket(5000) == 4
    assert bucket(LATENCY_BUCKETS_MS[-1]) == 4
    assert bucket("+Inf") == 5      # equals the count
    assert 'hybrid_edge_latency_ms_count{region="local",edge_id="edge-local",source="cloud"} 5' in text
