"""Design review question 56: scripts/deploy.sh, run against stub CLIs."""

import json
import os
import pathlib
import stat
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = "123456789012.dkr.ecr.us-east-1.amazonaws.com/hybrid-edge/gateway"
EDGES = {"us-east-1": {"cluster": "hybrid-edge-us-east-1"}, "eu-west-1": {"cluster": "hybrid-edge-eu-west-1"}}


def stub(bin_dir, name, body):
    p = bin_dir / name
    p.write_text("#!/usr/bin/env bash\n" + f'echo "{name} $*" >> "$STUB_LOG"\n' + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def run(tmp_path, *args, missing_polls=0, never_replicate="", running_tag="v9"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    stub(bin_dir, "terraform", f"""
case "$*" in
  *"output -raw ecr_repository_url"*) echo "{REPO}" ;;
  *"output -json edges"*) echo '{json.dumps(EDGES)}' ;;
esac
""")
    # ECR: each region reports the tag missing for the first N polls.
    stub(bin_dir, "aws", f"""
case "$*" in
  *"ecr get-login-password"*) echo token ;;
  *"ecr describe-images"*)
    region=$(echo "$*" | sed -E 's/.*--region ([^ ]+).*/\\1/')
    [ "$region" = "{never_replicate}" ] && exit 1
    f="$STUB_STATE/polls-$region"; n=$(cat "$f" 2>/dev/null || echo 0); echo $((n+1)) > "$f"
    [ "$n" -ge {missing_polls} ] || exit 1 ;;
  *"ecs describe-services"*) echo "arn:aws:ecs:x:1:task-definition/edge:7" ;;
  *"ecs describe-task-definition"*)
    region=$(echo "$*" | sed -E 's/.*--region ([^ ]+).*/\\1/')
    if [ "$region" = "eu-west-1" ]; then echo "{REPO}:{running_tag}"; else echo "{REPO}:v9"; fi ;;
esac
""")
    stub(bin_dir, "docker", "cat >/dev/null 2>&1 || true\n")
    stub(bin_dir, "git", "echo deadbee\n")
    (tmp_path / "state").mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUB_LOG": str(log),
           "STUB_STATE": str(tmp_path / "state"), "POLL_S": "0", "REPLICATION_TIMEOUT_S": "2"}
    p = subprocess.run(["bash", str(ROOT / "scripts/deploy.sh"), *args], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=60)
    return p, log.read_text() if log.exists() else ""


def test_happy_path_waits_for_replication_then_verifies_every_region(tmp_path):
    p, calls = run(tmp_path, "--yes", "v9", missing_polls=2)
    assert p.returncode == 0, p.stderr
    assert "deploy of v9 complete in: eu-west-1 us-east-1" in p.stdout
    assert calls.count("ecr describe-images --region eu-west-1") == 3  # polled until replicated
    apply = [c for c in calls.splitlines() if c.startswith("terraform") and " apply" in c]
    assert apply == ["terraform -chdir=%s apply -auto-approve -var gateway_image_tag=v9" % (ROOT / "infra/terraform")]
    assert calls.index("ecr describe-images") < calls.index(" apply ")  # never apply before replication
    assert "ecs wait services-stable --region us-east-1 --cluster hybrid-edge-us-east-1 --services edge" in calls


def test_without_yes_apply_is_interactive(tmp_path):
    p, calls = run(tmp_path, "v9")
    assert p.returncode == 0, p.stderr
    assert "-auto-approve" not in calls and "apply -var gateway_image_tag=v9" in calls


def test_replication_timeout_fails_before_apply(tmp_path):
    p, calls = run(tmp_path, "--yes", "v9", never_replicate="eu-west-1")
    assert p.returncode == 1 and "did not replicate to eu-west-1" in p.stderr
    assert " apply " not in calls


def test_rolled_back_region_fails_the_deploy(tmp_path):
    p, _ = run(tmp_path, "--yes", "v9", running_tag="v8")  # eu-west-1 still on the old image
    assert p.returncode == 1
    assert "FAILED in: eu-west-1(running:v8)" in p.stderr and "us-east-1 runs v9" in p.stdout


def test_unknown_option_is_rejected(tmp_path):
    p, _ = run(tmp_path, "--force")
    assert p.returncode == 2 and "unknown option" in p.stderr
