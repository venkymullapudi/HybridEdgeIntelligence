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

1. **Cache and coalescing.** Repeated prompts are answered from an in-process TTL cache. Entries are scoped to the caller's API key, so one tenant never sees another's answers. Keys keep inner whitespace, so prompts that differ only in indentation (code, YAML) never share an answer. Identical requests that arrive at the same time share one computation (single-flight), so a burst of the same hard prompt makes one Claude call, not fifty.
2. **Pre-gate.** Requests go straight to Claude when the client sends `route: "cloud"` (if the policy allows overrides), or when the prompt leaves less than `edge_min_answer_tokens` of room in the edge model's context window.
3. **Edge generation, streamed.** vLLM streams the answer with per-token logprobs and top-k alternatives, and the gate watches it as it arrives:
   * **Early abort.** If a window of tokens collapses below `gate_abort_window_prob`, generation stops and the request escalates right away instead of finishing a doomed answer. This only happens when Claude can actually take the request.
   * **Deadline.** If the edge model is slower than `edge_deadline_s` (for example, a queued GPU), the request escalates instead of waiting.
   * **Gray-zone self-consistency.** Extra samples are drawn only when the score is within `gate_consistency_band` of τ, which is the only time they can change the decision.
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
6. **Degrade, don't fail.** If Claude is down, rate-limited, over budget, or the circuit breaker is open, the edge answer is returned with `degraded: true`. A 503 is returned only when both tiers are down. A single `cloud_deadline_s` (default 90 s, below the ALB's 180 s idle timeout) caps each escalation, SDK retries included. After a cooldown the breaker lets **one** probe through (half-open) before it reopens to traffic. Budgets apply per node *and* per API key, so one noisy client can't drain the node.
7. **Shadow and record.** A small `shadow_rate` of edge-served requests is also sent to Claude **in the background**, without adding latency for the user. Escalations and shadow calls are written to Firehose → S3 (in-region by default) as `(prompt, edge answer, signals, Claude answer)`. Shadow calls matter because without them you would only have reference answers for requests the gate already rejected, and you couldn't calibrate the threshold. This data drives gate calibration and distillation into the edge model, so the escalation rate falls over time.
8. **Cost accounting and caching.** Escalations use prompt caching: the system prompt always, and the history too in multi-turn conversations (`cloud_prompt_caching`). Each response reports `cloud_cost_usd` (cache reads and writes priced separately) or `avoided_cost_usd`, and `/metrics` and CloudWatch keep running totals, so you can see what the edge is saving.

The design was stress-tested question by question. See [docs/DESIGN_REVIEW.md](docs/DESIGN_REVIEW.md) for what was found and fixed.

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
| `POST /v1/infer/stream` | The same request, as server-sent events. Edge and cached answers arrive as one `done` event. An escalation sends `escalating`, then `delta` events as Claude writes, then `done` (always authoritative, with `cloud_first_token_ms`). Failure ends with `error`. Idle gaps carry `: ping` heartbeats |
| `GET /healthz` | liveness check for the gateway process |
| `GET /readyz` | readiness: returns 503 when the local model is down, or isn't the configured `edge_model`. ALB and Global Accelerator use it to drain an edge |
| `GET /metrics` | Prometheus text: requests, escalations, escalation rate, degraded count, cloud-unavailable counts by reason, per-source latency histograms (for p95/p99), cloud tokens and cost. Needs an API key when keys are configured |

Authentication: `x-api-key: <key>` or `Authorization: Bearer <key>`, required when `HEI_API_KEYS` is set. Without configured keys, budgets and cache scope are per client address. That's the connection's address by default. Behind a proxy that appends `X-Forwarded-For` (the AWS ALB; the Terraform deployment sets this), set `HEI_TRUST_FORWARDED_FOR=true` to use the rightmost hop instead. Don't enable it without such a proxy, or clients can forge their identity.

Request IDs: send `X-Request-ID` (letters, digits, `.`, `_`, `-`; up to 64 characters) to correlate a request with gateway logs and escalation records. The gateway echoes it back, or generates one, in the response header and body.

Edge models without a system role (e.g. Gemma): set `HEI_EDGE_SYSTEM_ROLE=merge` (Terraform: `edge_system_role`) to fold the system prompt into the first user turn for the edge. Claude still gets a real system prompt.

Conversation shape: at most one `system` message, which must come first, the last message must be from the user (assistant prefill isn't supported), user and assistant messages can't be empty, and text must be valid Unicode (no lone surrogates). Anything else gets a 422. A large `max_tokens` shrinks the edge answer to fit the edge context window rather than forcing an escalation.

Request limits: requests over `max_request_chars` (default 100k) or `max_messages` (default 200) are rejected with 413. The raw body is also capped before it's read (`max_body_bytes`, by default derived from `max_request_chars`), so an oversized or endless upload can't fill gateway memory.

Client policy: set `HEI_ALLOW_CLIENT_OVERRIDES=false` to ignore `route` and `min_confidence` from callers. Even when overrides are allowed, `min_confidence` is capped at `max_client_min_confidence` (default 0.95), so a client can't force every request to Claude.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

No GPU: use the mock edge model, with escalation turned off:

```bash
HEI_EDGE_BACKEND=mock HEI_CLOUD_PROVIDER=none .venv/bin/hybrid-edge
```

To get real escalations, drop `HEI_CLOUD_PROVIDER=none`. Note that the Anthropic SDK finds credentials **on its own**: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login` profile. Without `HEI_CLOUD_PROVIDER=none`, hard prompts are sent to the real API and billed to whichever account it finds, even with no key exported.

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

The script waits until the image has replicated to every edge region, then shows the full Terraform plan for review (`--yes` skips the review, for CI). After applying, it checks that each region's ECS service actually runs the new tag, and exits non-zero naming any region the deployment circuit breaker rolled back.

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
* **Bedrock instead of an API key:** set `cloud_provider = "bedrock"`. Tasks then call Claude in their own region through IAM. If the model isn't offered in an edge's region, point that edge at a nearby one with `bedrock_region_overrides`. Before you rely on this, check that the model is enabled for your account in every edge region, and that the IAM actions in `modules/edge_region/ecs.tf` match what the Bedrock endpoint you use requires.
* **TLS:** pass `certificate_arns` (one ACM cert per edge region) for HTTPS on 443. Without them, `terraform plan` stops with an error unless you set `allow_insecure_http = true`, which sends client API keys in cleartext and is only suitable for a demo.
* **Data residency:** `records_residency = "regional"` (the default) keeps each edge's prompts and answers in a bucket in that edge's own region. `"central"` sends them all to one bucket in `home_region`.
* **Shadow traffic:** `shadow_rate` (default 0.01) costs roughly that fraction of edge-served traffic in extra Claude calls, capped by `shadow_max_per_min`. Set it to 0 to turn shadowing off. It's ignored, with a startup warning, unless records are kept with their text, since otherwise nothing could be calibrated from them.
* **Debugging:** ECS Exec into edge containers is off by default, since they hold the Anthropic key. Set `enable_execute_command = true` to allow it. Tasks can't reach instance metadata, so they never get the GPU node's credentials.
* **Alarms:** each edge alarms on escalation rate and degraded rate, and sends notifications to a regional SNS topic. Set `alarm_emails` to subscribe people, who must confirm AWS's subscription email.
* **Cost:** every edge region runs at least one GPU node (g6.xlarge by default) plus a NAT gateway, around the clock. ECS always keeps at least `desired_tasks` (≥ 1) running, and an edge with no task gets no traffic, so it could never scale up again. Lowering `min_nodes` doesn't change that floor. To stop paying for a region, remove it from `edge_regions`: Global Accelerator sends its users to the nearest remaining edge.
* **Model choice:** Qwen2.5-7B-Instruct (Apache-2.0) fits on a single L4. For gated models such as Llama, set `enable_hf_token = true` and store your Hugging Face token.
* The ECR replication configuration is account-wide. Set `manage_ecr_replication = false` if you already manage it somewhere else.

## Calibrating the gate

Choosing τ trades **escalation rate** (cost) against **edge precision** (quality). Once records are flowing:

1. Collect records that include **shadow** calls (`shadow_rate > 0`). Without them you only have labels for answers the gate already rejected.
2. Copy the records down and label them with the Claude judge. Both tools accept files, Firehose's gzip objects and whole directories, and they skip (and count) malformed lines. The judge decides whether each edge answer was acceptable using Claude's answer as the reference. Every judged record is a billed call, and `--limit` caps the spend. Human labels in a `label` field work too.

   ```bash
   aws s3 sync s3://<records-bucket>/escalations/ data/escalations/
   hybrid-edge-label data/escalations/ -o labelled.jsonl --limit 5000
   ```

   Progress is checkpointed (`--checkpoint-every`, default 100) and saved if the run is interrupted. To resume without re-paying for finished labels, re-run with the output file as input.

   As a crude fallback, `hybrid-edge-calibrate --weak-labels` labels by lexical overlap instead.
   Shadow records carry `sample_weight = 1/shadow_rate`, and the calibrator uses it, so its precision and coverage numbers describe real traffic rather than the sample.
3. Fit the calibrator:

   ```bash
   hybrid-edge-calibrate labelled.jsonl -o calibration.json --target-precision 0.95 --sweep
   ```

   `--sweep` adds coverage, precision and escalation rate at every threshold, so you can pick the target precision from the curve.

   ```
   {"rows": 5000, "threshold": 0.62, "edge_coverage": 0.78, "edge_precision": 0.951, "escalation_rate": 0.22}
   ```
4. Ship it. Upload `calibration.json` to S3 under a new key per version, and set Terraform's `gate_calibration_s3_uri` (e.g. `s3://my-bucket/gate/calibration-v3.json`). Edges load it at startup, with read access to only that object. Locally, `HEI_GATE_CALIBRATION_PATH` takes a file path or the same `s3://` URI. The calibration's threshold replaces `gate_threshold`. The file records the edge model it was fit on, and the gateway refuses to start with a different `edge_model`, so recalibrate after every model change. If your records mix models, pass `--edge-model` to `hybrid-edge-calibrate`.

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
  calibrate.py   fits the gate from labelled records
  label.py       Claude judge that labels records for calibration
docs/            design review (questions, findings, fixes)
infra/terraform/ multi-region AWS deployment
docker/          gateway image and local compose setup
tests/           unit and integration tests (mock edge model, fake Claude client)
```
