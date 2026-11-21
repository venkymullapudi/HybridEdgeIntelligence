# Design review: grilling v0.1

Each question below was put to the v0.1 design. Each answer says what was wrong
and what v0.2 changes.

## 1. Can you actually calibrate the gate from the data you log?

**No. The labels were biased by how the data was selected.** Claude only saw
requests the gate had already rejected, so a confident edge answer was never
paired with a reference answer. A calibrator trained on that data learns
nothing about the region the threshold is supposed to protect: answers that
scored above τ and were wrong.

**Fix:** *shadow escalation.* A configurable fraction (`shadow_rate`) of
requests the edge serves is also sent to Claude **in the background**. The user
gets the edge answer with no extra latency, and the record stores both answers.
Shadow calls use their own small budget, so they can't crowd out real
escalations.

## 2. Are lexical-overlap labels good enough?

**No.** "Paris" and "The capital is Paris." overlap poorly, while "option A"
and "option B" overlap well. Jaccard labels are noise.

**Fix:** `hybrid-edge-label`, a judge that asks Claude, using structured
output, whether the edge answer is acceptable given the reference answer. It
writes a `label` field that `hybrid-edge-calibrate` uses. Weak labels are kept
only as a fallback.

## 3. What does an escalated request cost in latency?

**Too much.** The user waits for the *full* edge generation, then the *full*
Claude call. A doomed edge answer still runs to `max_tokens`.

**Fixes:**
* **Early abort.** The edge answer is streamed and scored as it arrives. If a
  window of tokens falls below `gate_abort_window_prob` (after
  `gate_abort_min_tokens`), generation stops and the request escalates
  immediately.
* **Edge deadline.** If the edge model is queued or slow past
  `edge_deadline_s`, the request is cancelled and escalated rather than waiting.

## 4. Self-consistency doubles GPU load. Is that worth it on every request?

**No.** Extra samples only change the decision when the primary score is close
to τ.

**Fix:** *gray-zone sampling.* Extra samples are drawn only when
`|score − τ| < gate_consistency_band`. Clear accepts and clear rejects skip
them.

## 5. What happens when 50 users send the same hard prompt at once?

**50 Claude calls** (a thundering herd), because the cache is filled only
after the first call returns.

**Fix:** *single-flight.* Concurrent identical requests share one in-flight
computation.

## 6. Does the circuit breaker really go half-open?

**No. It had a bug.** After the cooldown, `open` returned `False` for
*everyone*, so a full burst hit a cloud that might still be down.

**Fix:** a proper half-open state. Exactly one probe is allowed through. If it
succeeds the breaker closes; if it fails the breaker re-opens.

## 7. Can one client run up the bill?

**Yes.** Any caller could send `route: "cloud"` or `min_confidence: 1.0`, and
one noisy client could drain the node's whole escalation budget.

**Fixes:**
* A **per-API-key escalation budget** (`cloud_max_escalations_per_key_per_min`)
  on top of the per-node budget.
* `allow_client_overrides` (default on) can turn off `route` and
  `min_confidence` overrides. A per-request `min_confidence` can only *raise*
  confidence requirements up to `max_client_min_confidence`.

## 8. Can the cache leak one tenant's answer to another?

**Yes**, when two tenants send the same messages but expect different answers,
for example because of tenant-specific system prompts or data. The key covered
only the messages.

**Fix:** the cache key includes a hash of the caller's API key. Sharing across
tenants is opt-in (`cache_shared_across_keys`).

## 9. Where does EU user data go?

**To us-east-1.** Every edge's Firehose wrote prompts to one central bucket,
which is a cross-border transfer that many teams can't accept.

**Fix:** `records_residency = "regional"` is the new Terraform default: each
edge writes to a bucket in its own region. `"central"` is still available.

## 10. Are API keys sent in cleartext?

**By default, yes.** The default was HTTP on port 80 behind Global Accelerator.

**Fix:** Terraform now refuses to deploy without certificates unless
`allow_insecure_http = true` is set explicitly.

## 11. How do you show this is saving money?

**v0.1 couldn't.** There was no notion of cost.

**Fix:** the gateway estimates the dollar cost of each Claude call, and the
cost *avoided* by serving at the edge, using configurable per-MTok prices. Both
appear in the response `usage`, in `/metrics` and in EMF metrics
(`CloudCostUsd`, `AvoidedCostUsd`).

## 12. Does it run on the SDK version the Docker image installs?

**Unverified in v0.1.** Tests ran on Python 3.9 with `anthropic` 0.x, but the
image (Python 3.12) installs 1.x.

**Fix:** tests now run on both, locally and in CI.

## 13. What stops a client sending a 5 MB prompt?

**Nothing in v0.2.** An oversized request used gateway memory, and because it
overflowed the edge context window, it went *straight to Claude*. The client
chose the size of the bill.

**Fix:** the gateway rejects requests over `max_request_chars` (default 100k)
or `max_messages` (default 200) with HTTP 413, before any model sees them.

## 14. What happens to records under load?

**They piled up without bound.** Each record was handed to the default thread
pool and written with one Firehose `PutRecord` call per record. In a traffic
spike the executor queue grew without limit, and each call paid a full round
trip.

**Fix:** a bounded queue (`record_queue_size`) drained by one background thread
that writes batches. Firehose uses `PutRecordBatch` (up to 500 records), and
records it rejects are retried with backoff. When the queue is full, records
are **dropped and counted** (`records_dropped_total`) rather than slowing
serving. Shutdown flushes the queue. Oversized records (over 1000 KB) are
dropped too, since Firehose would reject them.

## Not changed (and why)

* **Shared cache across nodes (Redis/ElastiCache):** it adds a dependency and
  network hops to every request. The per-node cache plus single-flight covers
  the hot path. Revisit if the hit rate matters at scale.
* **Global (cross-node) budget:** this needs shared state (DynamoDB or Redis).
  Per-node and per-key buckets cap the damage, and the escalation-rate alarm
  catches drift.
* **Streaming responses to clients:** the gate has to decide before the user
  sees any tokens, or the edge answer would have to be retracted mid-stream.
  Streaming Claude's answer after an escalation is a reasonable next step.
