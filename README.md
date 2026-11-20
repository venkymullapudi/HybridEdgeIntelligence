# Hybrid Edge Intelligence

An open-weight model answers every request at the AWS edge region nearest the
user. A **confidence gate** scores each answer, and only low-confidence
requests escalate to a cloud LLM (Claude). Most traffic stays cheap and local,
and hard requests still get frontier-model quality.

```
                       ┌──────────────────────────────────────────────┐
  client ──anycast──▶  │  AWS Global Accelerator  (nearest healthy edge)│
                       └───────┬───────────────┬───────────────┬───────┘
                               ▼               ▼               ▼
                          us-east-1        eu-west-1       ap-south-1   … more edges
                         ┌─────────────────────────────────────────┐
                         │ ALB → ECS task on a GPU node (g6.xlarge)  │
                         │  ┌───────────┐  localhost  ┌──────────┐ │
                         │  │  gateway  │ ──────────▶ │  vLLM    │ │
                         │  │ (gate)    │ ◀─logprobs─ │ Qwen 7B  │ │
                         │  └─────┬─────┘             └──────────┘ │
                         └────────┼────────────────────────────────┘
                     score < τ    │  (only then)
                                  ▼
                        Claude (Claude API or Amazon Bedrock)
                                  │
       escalation records ──▶ Firehose ──▶ central S3  ──▶ gate calibration / distillation
```

## How a request flows

1. **Cache.** Repeated prompts are answered from an in-process TTL cache. Cached Claude answers are the most valuable hits.
2. **Pre-gate.** Requests go straight to Claude when the client sends `route: "cloud"` or the prompt won't fit the edge model's context window.
3. **Edge generation.** vLLM returns the answer along with per-token logprobs and top-k alternatives. You can also request *k* extra samples for self-consistency.
4. **Confidence gate.** The gate scores the answer in `[0, 1]` from these signals:
   | signal | catches |
   |---|---|
   | `mean_prob`: geometric-mean token probability | general uncertainty |
   | `min_window_prob`: worst sliding window of tokens | one guessed fact (a number or a name) inside fluent text |
   | `entropy_conf`: 1 − normalised top-k entropy | flat next-token distributions |
   | `consistency`: agreement across samples (optional) | unstable answers |
   | hedge phrases, truncation (penalties) | "I'm not sure…", answers cut off by `max_tokens` |

   A hand-weighted score is used out of the box. Once you have logged data, a
   **calibrated logistic model** replaces it (see *Calibrating the gate* below).
5. **Escalate or serve.** If `score ≥ τ`, the edge answer is returned. Otherwise the request goes to Claude (`claude-opus-5-5`, adaptive thinking, effort `medium` by default). Claude's refusal fallbacks are enabled: server-side `fallbacks: "default"` on the Claude API, and the SDK's client-side middleware on Bedrock.
6. **Degrade, don't fail.** If Claude is down, rate-limited, over budget, or the circuit breaker is open, the edge answer is returned with `degraded: true`. A 503 is returned only when both tiers are down.
7. **Record.** Every escalation, plus a small sample of confident requests, is written to Firehose → S3 as `(prompt, edge answer, signals, Claude answer)`. This data drives gate calibration and distillation into the edge model, so the escalation rate falls over time.

Each response tells you what happened:

```json
{
  "answer": "…",
  "source": "edge | cloud | cache",
  "escalated": true,
  "degraded": false,
  "confidence": 0.41,
  "threshold": 0.7,
  "reasons": ["score 0.410 < threshold 0.700", "edge answer hedges"],
  "region": "ap-south-1",
  "model": "claude-opus-5-5",
  "latency_ms": 2140.3,
  "usage": {"edge_completion_tokens": 52, "cloud_input_tokens": 61, "cloud_output_tokens": 180}
}
```

## API

| | |
|---|---|
| `POST /v1/infer` | `{"prompt": "…"}` or `{"messages": [{role, content}…]}`. Optional fields: `route` (`auto`, `edge` or `cloud`), `min_confidence` (overrides τ for this request), `max_tokens`, `use_cache`, `include_signals` |
| `GET /healthz` | liveness check for the gateway process |
| `GET /readyz` | readiness: returns 503 when the local model is down. ALB and Global Accelerator use it to drain an edge |
| `GET /metrics` | Prometheus text: requests, escalations, escalation rate, degraded count, latency and cloud tokens |

Authentication: `x-api-key: <key>` or `Authorization: Bearer <key>`, required when `HEI_API_KEYS` is set.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

No GPU: use the mock edge model. Set `ANTHROPIC_API_KEY` to get real escalations, or use `HEI_CLOUD_PROVIDER=none`.

```bash
HEI_EDGE_BACKEND=mock .venv/bin/hybrid-edge
```

```bash
curl -s localhost:8080/v1/infer -H 'content-type: application/json' -d '{"prompt":"prove this hard theorem","include_signals":true}'
```

With a real open-weight model, run any OpenAI-compatible server that returns logprobs, such as vLLM, llama.cpp `server` or Ollama:

```bash
docker compose -f docker/docker-compose.yml --profile gpu up
```

All settings are listed in [`hybrid_edge/config.py`](hybrid_edge/config.py). Set any of them as `HEI_<NAME>`, or put them in a YAML file (see [`config.example.yaml`](config.example.yaml)) and point `HEI_CONFIG` at it.

## Deploy to AWS

[`infra/terraform`](infra/terraform) creates:

* **Global:** a Global Accelerator (anycast IPs, nearest-healthy-region routing, automatic failover); an ECR repo replicated to every edge region; Secrets Manager secrets replicated to every edge region; and an encrypted S3 bucket for escalation records.
* **Each edge region** ([`modules/edge_region`](infra/terraform/modules/edge_region)): a VPC (public ALB and private GPU subnets, NAT, S3 endpoint), and an ECS cluster on GPU instances (ECS GPU AMI, ASG and capacity provider). Each task runs the **vLLM and gateway containers** side by side. The module also sets up request-based autoscaling, a Firehose stream to the central bucket, CloudWatch logs and EMF metrics, and an alarm on the escalation rate.

Edge regions included: `us-east-1`, `eu-west-1`, `ap-south-1`, `ap-southeast-1`. Choose which ones to deploy with `edge_regions`. To add a region, add a provider alias in `versions.tf` and copy one module block in `edges.tf`.

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars    # edit regions, model, instance type
terraform init
```

Create the registry first, so the image exists before any ECS service starts:

```bash
terraform apply -target=aws_ecr_repository.gateway -target=aws_ecr_replication_configuration.this
```

From the repo root, build the gateway image, push it, and apply everything:

```bash
scripts/deploy.sh v1
```

Put the secret values in through the AWS CLI so they never land in Terraform state. Do this for each name in the `secrets_to_populate` output:

```bash
aws secretsmanager put-secret-value --secret-id hybrid-edge/anthropic-api-key --secret-string "$ANTHROPIC_API_KEY"
```

```bash
aws secretsmanager put-secret-value --secret-id hybrid-edge/gateway-api-keys --secret-string "key1,key2"
```

Then call the edge through the single global endpoint:

```bash
curl -s "http://$(terraform -chdir=infra/terraform output -raw global_endpoint_dns)/v1/infer" -H "x-api-key: key1" -H 'content-type: application/json' -d '{"prompt":"hello"}'
```

Notes:
* **Bedrock instead of an API key:** set `cloud_provider = "bedrock"`. Tasks then call Claude in their own region through IAM. Before you rely on this, check that the model is enabled for your account in every edge region, and that the IAM actions in `modules/edge_region/ecs.tf` match what the Bedrock endpoint you use requires.
* **TLS:** pass `certificate_arns` (one ACM cert per region) for HTTPS on 443. The default is HTTP on port 80 only, which suits a demo but not production.
* **Cost:** each edge runs at least one g6.xlarge GPU node plus a NAT gateway. Set `min_nodes = 0` on regions you only want as burst capacity.
* **Model choice:** Qwen2.5-7B-Instruct (Apache-2.0) fits on a single L4. For gated models such as Llama, set `enable_hf_token = true` and store your Hugging Face token.
* The ECR replication configuration is account-wide. Set `manage_ecr_replication = false` if you already manage it somewhere else.

## Calibrating the gate

Choosing τ trades **escalation rate** (cost) against **edge precision** (quality). Once records are flowing:

1. Label each record (`label`: 1 if the edge answer was acceptable, 0 if not). You can do this by human review or with an offline LLM judge comparing `edge_answer` with `cloud_answer`. For a quick start, `--weak-labels` labels by lexical agreement between the two answers.
2. Fit the calibrator:

   ```bash
   hybrid-edge-calibrate records.jsonl -o calibration.json --target-precision 0.95
   ```

   ```
   {"rows": 5000, "threshold": 0.62, "edge_coverage": 0.78, "edge_precision": 0.951, "escalation_rate": 0.22}
   ```
3. Ship `calibration.json` with the gateway and set `HEI_GATE_CALIBRATION_PATH`. Its threshold replaces `gate_threshold`.

The same records double as a distillation set: fine-tune the edge model on `(messages, cloud_answer)` pairs, redeploy it, and recalibrate.

## Layout

```
hybrid_edge/
  backends/      edge model clients: OpenAI-compatible (vLLM etc.) and a mock
  gate/          uncertainty signals and the confidence gate (heuristic or calibrated)
  cloud/         Claude escalation: refusal fallbacks, circuit breaker, budget
  orchestrator.py  request flow: cache → edge → gate → escalate or degrade
  server.py      FastAPI gateway
  telemetry.py   CloudWatch EMF metrics, Prometheus /metrics, Firehose record sink
  calibrate.py   fits the gate from logged records
infra/terraform/ multi-region AWS deployment
docker/          gateway image and local compose setup
tests/           unit and integration tests (mock edge model, fake Claude client)
```
