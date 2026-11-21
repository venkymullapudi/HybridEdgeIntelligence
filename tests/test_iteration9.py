"""Design review questions 32-33."""

import json

from fastapi.testclient import TestClient

from hybrid_edge.calibrate import fit, main as calibrate_main, sweep
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.orchestrator import caller_id
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink
from .test_misc import _synthetic


# --- Q32: unauthenticated callers can't mint budgets --------------------------
def _client(settings):
    settings.cloud_max_escalations_per_key_per_min = 1
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    return TestClient(create_app(settings, o)), fake


def test_random_keys_do_not_bypass_per_caller_budget_without_api_keys(settings):
    c, fake = _client(settings)
    with c:
        sources = [c.post("/v1/infer", json={"prompt": f"hard {i}", "use_cache": False},
                          headers={"x-api-key": f"random-{i}"}).json()["source"] for i in range(3)]
    assert sources == ["cloud", "edge", "edge"]  # one budget for this client, not one per made-up key
    assert len(fake.calls) == 1


def test_unauthenticated_identity_uses_alb_appended_hop(settings):
    c, fake = _client(settings)
    with c:
        # The client forges the left entries; the ALB appends the real address last.
        a = c.post("/v1/infer", json={"prompt": "hard a", "use_cache": False},
                   headers={"x-forwarded-for": "1.1.1.1, 203.0.113.7"}).json()
        b = c.post("/v1/infer", json={"prompt": "hard b", "use_cache": False},
                   headers={"x-forwarded-for": "9.9.9.9, 203.0.113.7"}).json()
        other = c.post("/v1/infer", json={"prompt": "hard c", "use_cache": False},
                       headers={"x-forwarded-for": "198.51.100.2"}).json()
    assert a["source"] == "cloud" and b["source"] == "edge"  # same real client
    assert other["source"] == "cloud"                         # a different client has its own budget


def test_forwarded_for_ignored_when_untrusted(settings):
    settings.trust_forwarded_for = False
    c, fake = _client(settings)
    with c:
        a = c.post("/v1/infer", json={"prompt": "hard a", "use_cache": False}, headers={"x-forwarded-for": "1.1.1.1"})
        b = c.post("/v1/infer", json={"prompt": "hard b", "use_cache": False}, headers={"x-forwarded-for": "2.2.2.2"})
    assert a.json()["source"] == "cloud" and b.json()["source"] == "edge"


def test_configured_keys_still_identify_callers(settings):
    settings.api_keys = ["k1", "k2"]
    c, fake = _client(settings)
    with c:
        a = c.post("/v1/infer", json={"prompt": "hard a", "use_cache": False}, headers={"x-api-key": "k1"}).json()
        b = c.post("/v1/infer", json={"prompt": "hard b", "use_cache": False}, headers={"x-api-key": "k2"}).json()
    assert a["source"] == b["source"] == "cloud"
    assert caller_id("k1") != caller_id("ip:testclient")


# --- Q33: threshold sweep -----------------------------------------------------
def test_sweep_is_monotone_and_matches_endpoints():
    rows = _synthetic(300)
    cal = fit(rows, epochs=200)
    table = sweep(cal, rows, step=0.1)
    assert table[0]["threshold"] == 0.0 and table[0]["edge_coverage"] == 1.0
    covs = [r["edge_coverage"] for r in table]
    assert covs == sorted(covs, reverse=True)  # raising the threshold never serves more at the edge
    assert all(abs(r["edge_coverage"] + r["escalation_rate"] - 1) < 1e-3 for r in table)


def test_calibrate_cli_sweep(tmp_path, capsys):
    rec = tmp_path / "r.jsonl"
    with open(rec, "w") as fh:
        for feats, y in _synthetic(200):
            fh.write(json.dumps({"features": feats, "label": y}) + "\n")
    calibrate_main([str(rec), "-o", str(tmp_path / "c.json"), "--sweep"])
    report = json.loads(capsys.readouterr().out)
    assert len(report["sweep"]) == 21 and report["sweep"][-1]["threshold"] == 1.0
