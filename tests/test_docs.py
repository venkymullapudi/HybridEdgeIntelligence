"""Design review question 72: documented names must exist in the code.

Doc drift caused real bugs twice (questions 59 and 60). Every allowlist here is
derived from the code, so renaming a setting or mistyping one in the docs fails CI.
"""

import dataclasses
import pathlib
import re

import yaml

from hybrid_edge.config import Settings
from hybrid_edge.gate.signals import Signals
from hybrid_edge.server import InferRequest, InferResponse

ROOT = pathlib.Path(__file__).resolve().parents[1]
TF = ROOT / "infra" / "terraform"
SETTINGS = {f.name for f in dataclasses.fields(Settings)}


def tf_names(kind):
    text = "".join(p.read_text() for p in TF.rglob("*.tf"))
    return set(re.findall(r'%s "([a-z0-9_]+)"' % kind, text))


def known_names():
    usage_keys = set(re.findall(r'"((?:edge|cloud|avoided)_[a-z_]+)"',
                                (ROOT / "hybrid_edge" / "orchestrator.py").read_text()))
    return (SETTINGS | tf_names("variable") | tf_names("output") | set(InferRequest.model_fields)
            | set(InferResponse.model_fields) | {f.name for f in dataclasses.fields(Signals)}
            | set(Signals().vector()) | usage_keys)


def test_every_hei_env_var_is_a_setting():
    files = [ROOT / "README.md", ROOT / "docs" / "DESIGN_REVIEW.md", ROOT / "docker" / "docker-compose.yml",
             ROOT / ".github" / "workflows" / "ci.yml", *TF.rglob("*.tf")]
    bad = [f"{p.relative_to(ROOT)}: HEI_{n}" for p in files
           for n in sorted(set(re.findall(r"\bHEI_([A-Z0-9_]+)\b", p.read_text())))
           if n != "CONFIG" and n.lower() not in SETTINGS]
    assert bad == [], f"documented env vars that aren't settings: {bad}"


def test_example_config_only_uses_real_settings():
    cfg = yaml.safe_load((ROOT / "config.example.yaml").read_text())
    assert set(cfg) <= SETTINGS, set(cfg) - SETTINGS
    Settings.from_dict(cfg)  # and it validates


def test_readme_names_exist():
    readme = (ROOT / "README.md").read_text()
    names = set(re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", readme))
    unknown = sorted(names - known_names())
    assert unknown == [], f"README mentions names the code doesn't define: {unknown}"


def test_checker_catches_a_typo():
    assert "gate_treshold" not in known_names() and "gate_threshold" in known_names()
