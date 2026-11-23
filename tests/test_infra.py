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


def test_tasks_cannot_reach_instance_metadata():
    """Containers must not borrow the GPU node's role via IMDS."""
    ecs = (TF / "modules/edge_region/ecs.tf").read_text()
    assert "ECS_AWSVPC_BLOCK_IMDS=true" in ecs
    assert re.search(r"http_put_response_hop_limit\s*=\s*1\b", ecs)
    assert re.search(r'http_tokens\s*=\s*"required"', ecs)  # IMDSv2 only


def test_ecs_exec_is_opt_in_and_works_when_enabled():
    ecs = (TF / "modules/edge_region/ecs.tf").read_text()
    assert "enable_execute_command = var.enable_execute_command" in ecs
    assert re.search(r'variable "enable_execute_command" \{[^}]*default\s*=\s*false',
                     (TF / "variables.tf").read_text(), re.S)
    assert "var.enable_execute_command ? [{" in ecs and "ssmmessages:OpenDataChannel" in ecs


def test_every_records_bucket_requires_tls():
    buckets = list(blocks("aws_s3_bucket"))
    policies = list(blocks("aws_s3_bucket_policy"))
    assert {f for f, n, _ in buckets if n == "records"} == {"main.tf", "telemetry.tf"}
    for f, _, _ in [b for b in buckets if b[1] == "records"]:
        tls = [body for pf, n, body in policies if pf == f and n == "records_tls_only"]
        assert tls and '"aws:SecureTransport" = "false"' in tls[0], f"{f}: records bucket allows plain HTTP"


def test_rolling_deploy_has_a_spare_gpu_at_full_scale():
    """Service tasks may reach max_nodes; the ASG must allow one more node so a
    rolling deploy (new task before old stops, one GPU each) can always place."""
    ecs = (TF / "modules/edge_region/ecs.tf").read_text()
    assert re.search(r"max_size\s*=\s*var\.max_nodes \+ 1", ecs)
    assert re.search(r"max_capacity\s*=\s*var\.max_nodes\b", ecs)
    assert "minimum_healthy_percent" not in ecs or "deployment_minimum_healthy_percent = 100" in ecs


def test_global_accelerator_fails_over_quickly():
    edges = (TF / "edges.tf").read_text()
    assert re.search(r"health_check_interval_seconds\s*=\s*10\b", edges)


def test_scaling_settings_are_checked_at_plan_time():
    asg = next(body for f, n, body in blocks("aws_autoscaling_group") if n == "gpu")
    assert "var.desired_tasks >= 1 && var.desired_tasks <= var.max_nodes" in asg
    assert "var.min_nodes >= 0 && var.min_nodes <= var.max_nodes" in asg


def test_readme_does_not_promise_scale_to_zero():
    readme = (TF.parents[1] / "README.md").read_text()
    assert "min_nodes = 0" not in readme


def test_image_data_dir_is_writable_by_the_runtime_user():
    """The file sink defaults to data/escalations.jsonl under root-owned /app; the image
    runs as uid 10001, so that directory must be created and handed to it."""
    dockerfile = (TF.parents[1] / "docker" / "Dockerfile").read_text()
    assert "mkdir -p /app/data && chown app:app /app/data" in dockerfile
    assert dockerfile.index("chown app:app /app/data") < dockerfile.index("USER app")
    from hybrid_edge.config import Settings
    assert Settings().record_path == "data/escalations.jsonl"  # the default the image must support


def test_gateway_runs_with_least_privilege():
    ecs = (TF / "modules/edge_region/ecs.tf").read_text()
    gw = ecs[ecs.index('name              = "gateway"'):]
    gw = gw[:gw.index("logConfiguration")]
    assert 'user                   = "10001"' in gw
    assert "readonlyRootFilesystem = !var.enable_execute_command" in gw  # read-only unless Exec is on
    assert 'capabilities       = { drop = ["ALL"] }' in gw
    ci = (TF.parents[1] / ".github/workflows/ci.yml").read_text()
    assert "--read-only --cap-drop ALL" in ci  # and CI proves the image runs that way


DIGEST_RE = r"@sha256:[0-9a-f]{64}"


def test_vllm_image_is_pinned_by_digest_everywhere():
    for p in [TF / "variables.tf", TF / "modules/edge_region/variables.tf"]:
        block = re.search(r'variable "vllm_image" \{.*?\n\}', p.read_text(), re.S).group(0)
        assert re.search(r'default\s*=\s*"vllm/vllm-openai' + DIGEST_RE, block), p.name
    compose = (TF.parents[1] / "docker" / "docker-compose.yml").read_text()
    assert re.search(r"image: vllm/vllm-openai:v[\d.]+" + DIGEST_RE, compose)


def test_gateway_tags_are_immutable_and_never_latest():
    assert 'image_tag_mutability = "IMMUTABLE"' in (TF / "main.tf").read_text()
    var = re.search(r'variable "gateway_image_tag" \{.*?\n\}', (TF / "variables.tf").read_text(), re.S).group(0)
    assert 'default     = ""' in var and 'var.gateway_image_tag != "latest"' in var
    td = next(body for f, n, body in blocks("aws_ecs_task_definition") if n == "edge")
    assert '!endswith(var.gateway_image, ":")' in td
    assert ":latest" not in (TF.parents[1] / "scripts" / "deploy.sh").read_text()
