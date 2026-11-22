"""Static guards on the Terraform that tofu validate can't express."""

import pathlib
import re

TF = pathlib.Path(__file__).resolve().parents[1] / "infra" / "terraform"


def blocks(kind):
    for f in TF.rglob("*.tf"):
        text = f.read_text()
        for m in re.finditer(r'resource "%s" "(\w+)" \{' % kind, text):
            depth, i = 0, m.end() - 1
            for j in range(i, len(text)):
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                if depth == 0:
                    yield f.name, m.group(1), text[i:j + 1]
                    break


def test_every_alarm_notifies_someone():
    alarms = list(blocks("aws_cloudwatch_metric_alarm"))
    assert {n for _, n, _ in alarms} >= {"escalation_rate", "degraded_rate"}
    for f, name, body in alarms:
        assert "alarm_actions" in body, f"{f}: alarm {name} has no alarm_actions"


def test_alarm_topics_are_publishable_by_cloudwatch():
    for f, name, body in blocks("aws_sns_topic"):
        assert "alias/aws/sns" not in body, f"{f}: {name} uses aws/sns, which CloudWatch can't publish to"


def test_drain_timings_are_ordered():
    """cloud deadline < uvicorn graceful shutdown < ALB deregistration < ECS stopTimeout."""
    ecs = (TF / "modules/edge_region/ecs.tf").read_text()
    alb = (TF / "modules/edge_region/alb.tf").read_text()
    deadline = int(re.search(r'HEI_CLOUD_DEADLINE_S", value = "(\d+)"', ecs).group(1))
    dereg = int(re.search(r"deregistration_delay\s*=\s*(\d+)", alb).group(1))
    stop = int(re.search(r"stopTimeout\s*=\s*(\d+)", ecs).group(1))
    graceful = deadline + 5  # hybrid_edge.server.main
    assert deadline < graceful < dereg < stop


def test_alb_deployment_trusts_forwarded_for():
    """The app defaults to not trusting X-Forwarded-For; the ALB deployment must opt in,
    or every caller behind the load balancer would share the ALB's address."""
    ecs = (TF / "modules/edge_region/ecs.tf").read_text()
    assert '{ name = "HEI_TRUST_FORWARDED_FOR", value = "true" }' in ecs
