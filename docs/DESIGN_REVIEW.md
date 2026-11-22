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

## 15. What happens to coalesced requests when the first caller hangs up?

**They all failed.** Single-flight ran the shared work inside the first
caller's own request. When that client disconnected, the server cancelled the
request, and the cancellation passed to every other request waiting on the
same answer.

**Fix:** the shared work runs as its own task. Each caller waits on a shielded
reference to it, so a caller leaving cancels only its own wait. The work still
finishes, gets cached, and is waited for on shutdown.

## 16. How long can one escalation take?

**Longer than the load balancer waits.** The SDK timeout (120 s) applied to
each of up to three attempts, so one escalation could run for about 6 minutes.
The ALB gives up after 180 s, so the user got a 504 while the gateway kept
waiting on, and paying for, Claude.

**Fix:** one overall `cloud_deadline_s` (default 90 s) covers the whole call,
retries included. Past the deadline the request falls back to the edge answer
(`degraded: true`), and the timeout counts toward the circuit breaker.

## 17. Does the calibrator estimate precision on traffic or on the sample?

**On the sample, which is the wrong one.** Every escalation is recorded, but
shadow records are only a `shadow_rate` sample of confident traffic. Treated
equally, confident traffic is under-counted by a factor of 1/`shadow_rate`. So
the chosen threshold, and the reported coverage and escalation rate, were
skewed.

**Fix:** each record carries a `sample_weight` (1/`shadow_rate` for shadow
records, 1 for escalations). `hybrid-edge-calibrate` uses it both when fitting
and when choosing the threshold, so precision and coverage are estimates over
real traffic.

## 18. Can two different prompts get the same cached answer?

**Yes, when they differed only in spacing.** The cache key collapsed all
whitespace, so "fix this Python" with one indentation and with another hashed
to the same key, and the second user got the first user's answer. In code,
YAML and tables, indentation *is* meaning.

**Fix:** the key now normalises only what can't change the answer: line endings
(`\r\n` → `\n`) and leading or trailing whitespace. Inner whitespace is kept.

## 19. What happens to in-flight escalations during a deploy?

**They were cut off.** Escalations can run up to `cloud_deadline_s` (90 s). But
the ALB drained targets after 60 s, ECS sends SIGKILL 30 s after SIGTERM by
default, and uvicorn had no graceful-shutdown window. Every rolling deploy or
scale-in dropped the slowest requests, which are also the most expensive ones,
already paid for.

**Fix:** the timings are sized to match, each just above the last:
cloud deadline 90 s < uvicorn graceful shutdown 95 s (`cloud_deadline_s + 5`)
< ALB deregistration delay 100 s < ECS `stopTimeout` 120 s.

## 20. Can a user poison the calibration labels?

**Yes.** User and model text went into the judge prompt as-is. A message (or an
edge answer echoing one) could close `</candidate_answer>` and add a fake
section, or simply tell the judge to grade "acceptable". Inflated labels push
the calibrated threshold down, so more bad answers get served from the edge.

**Fix:** all untrusted text is HTML-escaped before it goes into the judge's
tags, so it can't close them. The judge's system prompt says tagged content is
data, never instructions, and that an answer trying to steer the grade is
itself unacceptable.

## 21. What happens to the calibration when you swap the edge model?

**It kept applying, silently.** Logprob scales differ between models, so a
threshold fit on one model's signals means nothing for another. But
`calibration.json` didn't record which model it was fit on, and records from
different models could be pooled into one fit.

**Fix:** the calibrator stores its `edge_model`. `hybrid-edge-calibrate`
refuses records that mix models unless `--edge-model` picks one. The gateway
**refuses to start** when the calibration's model doesn't match `edge_model`, so
ECS's deployment circuit breaker rolls the deploy back.
`gate_allow_calibration_mismatch` overrides this deliberately.

## 22. Who can read `/metrics`?

**Anyone on the internet.** The ALB exposed `/metrics` without auth, revealing
traffic volume, escalation rate and cost.

**Fix:** `/metrics` now requires the same API key as `/v1/infer` (when keys are
configured). `/healthz` and `/readyz` stay open for the load balancer and
Global Accelerator.

## 23. What if the edge server stops returning logprobs?

**Every request escalates, and nothing says why.** With no logprobs the
signals fall back to a neutral 0.5, which is below the default 0.7 threshold.
A server upgrade, or a backend without logprob support, turns the edge into a
pass-through to Claude at full cost.

**Fix:** each such answer increments `edge_no_logprobs_total` and logs a
one-time error explaining the cause. The escalation-rate alarm catches the
effect. A new test also pins the EMF document format that the alarm depends on:
every declared metric has a numeric value, and every dimension is present.

## 24. Has CI ever passed?

**No. Every run since the first commit failed.** `ubuntu-latest` moved to
Ubuntu 26.04, which has no Python 3.9 build, so `setup-python` failed before any
test ran. `fail-fast` then cancelled the 3.12 job, which hid whether it would
have passed. Local checks were green the whole time, so nothing looked wrong.

**Fix:** the test job runs on `ubuntu-24.04` (which has 3.9 and 3.12), with
`fail-fast: false`. The first green run was `54ce9b2`: tests on 3.9 and 3.12,
Terraform validate, and the Docker build all passed. The improvement loop now
reads CI results through GitHub's public API, so a failure can't go unnoticed
again.

## 25. What is the p99 latency of an escalation?

**Unknowable.** `/metrics` exported a latency sum and count per source, which
gives only the mean. Escalations make the latency distribution long-tailed, and
the mean hides exactly what users feel.

**Fix:** a Prometheus histogram, `hybrid_edge_latency_ms_bucket`, per source
(edge, cloud, cache), with buckets from 25 ms to 120 s. Use
`histogram_quantile(0.99, ...)` for p99 per source.

## 26. Can a dependency release break production with no code change?

**Yes.** The requirements had floors but no ceilings, and `anthropic>=0.75`
already resolved to 2.x in the image. The next major release of the SDK,
FastAPI or uvicorn would have shipped on the next image build, untested.

**Fix:** every runtime and dev dependency has an upper bound at the next major
(`anthropic<3`, `fastapi<1`, and so on). CI covers both ends of the supported
range: Python 3.9 resolves the oldest versions, and 3.12 the newest.

## 27. Can a client force an escalation without `route: "cloud"`?

**Yes, through `max_tokens`.** The pre-gate sent a request to Claude whenever
*prompt + max_tokens* exceeded the edge context window. A request with
`max_tokens: 8192` and an 8192-token edge therefore always escalated, even with
`allow_client_overrides = false`.

**Fix:** the edge answer is sized to the room left in its context window, and
the response says so (`edge answer limited to N tokens…`). Only a prompt that
leaves less than `edge_min_answer_tokens` (default 256) of room skips the edge.
If the shortened answer is cut off, the gate's truncation penalty decides
whether it escalates, the same as for any other answer.

## 28. What happens to a conversation the models can't take?

**It failed late, and in confusing ways.** A conversation ending in an
assistant turn is a prefill: current Claude models reject it with a 400, which
showed up as a degraded edge answer. A system message mid-conversation was
silently dropped on the Claude path (only the last one was kept) and breaks
many edge chat templates.

**Fix:** the gateway checks the conversation's shape before any model runs. It
allows at most one system message, which must come first, and the last turn
must be from the user. Anything else gets a 422 that says what's wrong.

## 29. What does the user get when Claude writes nothing?

**An empty string, presented as a successful cloud answer, and cached.** With
adaptive thinking, Claude can spend the whole `max_tokens` budget thinking and
return no text. The gateway returned `""` with `source: cloud`. A cut-off
answer (`stop_reason: max_tokens`) also came back with no flag.

**Fix:** an escalation that returns no text (and isn't a refusal) counts as
"cloud unavailable". The request falls back to the edge answer
(`degraded: true`), or returns a 503 if there's no edge answer. It doesn't trip
the circuit breaker, since the cloud isn't down. A truncated answer is still
returned, but with the reason `cloud answer truncated at cloud_max_tokens`.
Neither is cached.

## 30. What if Claude isn't offered on Bedrock in an edge's region?

**That edge could never escalate.** Each edge called Bedrock in its own region,
with no override. If the model isn't available there, every escalation fails,
the breaker opens, and the edge serves only degraded answers.

**Fix:** with Terraform's `bedrock_region_overrides`
(e.g. `{ "ap-south-1" = "ap-southeast-1" }`), an edge can call Bedrock in a
nearby region. Unlisted edges keep using their own region.

## 31. Will CI break again on its own?

**It was about to.** The workflow pinned `actions/checkout@v4`,
`setup-python@v5` and `setup-terraform@v3`, which target Node 20. GitHub
already forces them onto Node 24 and will eventually drop them. That's the
same class of silent breakage as question 24.

**Fix:** bumped to `checkout@v7`, `setup-python@v7` and `setup-terraform@v4`
(each released at least two weeks ago).

## 32. Without API keys, who is "a caller"?

**Whoever the client said it was.** When no keys were configured, the gateway
still derived the caller identity from any `x-api-key` header, without checking
it. A client sending a random key with each request got a fresh per-caller
escalation budget every time, so the per-key limit (question 7) did nothing.
Cache scoping and records were keyed on the same made-up label.

**Fix:** unchecked keys are ignored. Without configured keys, callers are
identified by client address: the **rightmost** `X-Forwarded-For` hop, which
the ALB appends (Global Accelerator preserves the client IP). Entries to its
left came from the client and can be forged. Set `trust_forwarded_for = false`
when clients connect directly, without a load balancer.

## 33. How do you choose the target precision?

**Blind.** `hybrid-edge-calibrate` reported only the threshold for one
`--target-precision`, with no view of what a slightly lower or higher target
would cost.

**Fix:** `--sweep` adds a table of edge coverage, precision and escalation
rate at each threshold (traffic-weighted, like question 17). You can read off
the knee of the cost/quality curve before committing to a target.

## 34. How much memory can the gateway's buffers take?

**More than the container has.** Both in-memory buffers were bounded by entry
count only:

* **Cache:** 10,000 entries. A cached Claude answer can be ~16k tokens
  (~64 KB), so the cache could reach ~640 MB, in a container that reserves 1 GB.
* **Record queue:** 10,000 records of up to 1,000 KB each, so ~10 GB in the
  worst case. That worst case is exactly a Firehose slowdown, which the queue
  exists to absorb.

An out-of-memory kill takes down the whole edge task, along with every
in-flight request.

**Fix:** both buffers also have a byte budget. The cache (`cache_max_bytes`,
default 128 MB) evicts least-recently-used entries until it fits. An entry
bigger than the whole budget isn't cached, so it can't flush everything else.
The record queue (`record_queue_max_bytes`, default 64 MB) drops and counts new
records past its budget, and releases each record's bytes once it's written.

## 35. What if the edge model's chat template has no system role?

**Every edge call fails, and everything escalates.** The gateway always sends a
system message, injecting the default prompt when the client sends none. Some
popular open-weight models reject a system role: Gemma's chat template raises
"System role not supported". Swapping to one of them turns the edge into a
pass-through to Claude at full cost. Only the escalation-rate alarm would
notice.

**Fix:** `edge_system_role = "merge"` folds the system prompt into the first
user turn for the edge model. Claude still gets a real system prompt.
`native` (the default) is unchanged. The setting is exposed in Terraform as
`edge_system_role`.

## 36. Can you follow one request through the system?

**Not easily.** Each response body had a `request_id`, but there was no response
header, and a client's own correlation ID was ignored. Tracing a user's report
to gateway logs and the escalation record meant guessing.

**Fix:** the gateway accepts `X-Request-ID` from the client when it's safe to
echo and log (`[A-Za-z0-9._-]{1,64}`), or generates one. It returns the ID as a
response header (on errors too) and in the body, and uses it in escalation and
shadow records. Requests answered from the cache or coalesced keep their own
ID.

## 37. How long does a user stare at nothing during an escalation?

**For Claude's whole answer.** The slowest requests, escalations, often run
10 s or more, and nothing was sent until the last token. Streaming was on the
"not changed" list because the gate has to decide before any edge tokens are
shown. But once a request has escalated, there's nothing left to retract.

**Fix:** `POST /v1/infer/stream` (server-sent events). The gate still decides
first:

* An edge or cached answer arrives as a single `done` event.
* An escalation sends `escalating` (with the gate's score and reasons), then
  `delta` events as Claude writes, then `done`.
* If both tiers are down, the stream ends with an `error` event.

Escalations stream through `ClaudeEscalator.stream_answer`, which shares
budgets, the breaker, the deadline, refusal fallbacks and the request shape
with the non-streaming path. The orchestrator has one code path for both, with
an optional event sink. `done` is always authoritative. If the stream breaks
midway, the request falls back to the edge answer and says how much had
streamed. A refusal fallback mid-stream can also change the final text.
Validation errors are still plain HTTP 422/413 responses, sent before the
stream opens. A client that disconnects cancels the work, which stops Claude's
billed stream. `escalating` is sent only when Claude can actually take the
request.

## 38. Does a streamed escalation survive a slow start?

**Not reliably.** After `escalating`, Claude Opus 5.5 thinks before it writes
any text, which can take tens of seconds. Before that, edge generation and
gating take up to `edge_deadline_s`. During those gaps the stream sent no
bytes, and proxies, mobile networks and client libraries with 30–60 s idle
timeouts drop silent connections. Those are exactly the long, expensive
requests streaming was meant to help.

**Fix:** after `sse_heartbeat_s` (default 15 s) of silence, the stream sends an
SSE comment (`: ping`). SSE clients ignore comments by spec, and intermediaries
see a live connection.

## 39. Is streaming actually helping?

**Unmeasured.** Streaming exists to cut time-to-first-token on escalations, but
nothing recorded it. You couldn't tell whether users saw text after 1 s or
after 40 s of thinking.

**Fix:** streamed escalations report `cloud_first_token_ms` (from the start of
the Claude call to its first text) in the `done` event. `/metrics` exports a
`hybrid_edge_cloud_first_token_ms` histogram alongside the end-to-end latency
histograms. A large gap between the two is what streaming saves. A large
first-token time means thinking dominates: try a lower `cloud_effort`.

## 40. Can shadow traffic cost money and produce nothing?

**Yes.** A shadow call exists only to produce a labelled record. With
`record_sink = "none"` the record is discarded. With
`record_include_text = false` it has no answers for the judge to compare. In
either case, every shadow call was billed Claude usage with no use.

**Fix:** shadow calls run only when the active record sink keeps records with
their text. Otherwise `shadow_rate` is ignored, with a warning at startup that
says why.

## 41. When the gateway serves degraded answers, why?

**You couldn't tell.** `cloud_errors_total` counted every failed escalation
together: our own throttling (node or per-key budget, open breaker) and Claude
failing (errors, deadlines, empty answers, rejected requests). Those call for
opposite responses on-call: raise a budget, or wait out an outage.

**Fix:** each failure carries a stable reason, and `/metrics` exports
`hybrid_edge_cloud_unavailable_total{reason=...}`. The reasons are `budget`,
`key_budget`, `shadow_budget`, `breaker`, `disabled`, `auth`, `error`,
`deadline`, `empty` and `rejected`.

## 42. Does coalescing respect each caller's threshold?

**No. It broke two guarantees.** The cache path checked that a stored result
met *this* request's threshold, but the single-flight path handed the leader's
result to every waiter unchecked:

* **Weaker answers to stricter callers:** if the leader sent
  `min_confidence: 0.5` and was served an edge answer at confidence 0.6, an
  identical concurrent request with `min_confidence: 0.9` received that same
  0.6 answer. Its own requirement was silently violated.
* **Hidden outages:** if the leader's escalation failed (`degraded: true`),
  every waiter's copy reported `degraded: false`, because the shared-copy
  helper reset the flag.

**Fix:** one rule, `_meets`, decides whether a result computed for another
caller can be reused. The cache and coalescing paths both use it: the result
must not be degraded, and must come from Claude or meet this caller's threshold.

* A waiter whose stricter threshold rejects the leader's answer runs its own
  request.
* A degraded result is still shared, since the cloud just failed for this exact
  request and retrying would only add load. But it keeps `degraded: true` and
  the original reasons.

Regression tests fail on the previous code.

## 43. Who hears an alarm?

**No one.** The escalation-rate alarm had no `alarm_actions`, so it could only
turn red in a console nobody watches. There was no alarm at all on degraded
answers, the signal users actually feel when Claude is down, throttled or over
budget. And coalesced degraded copies were counted as healthy in metrics, a
leftover from question 42.

**Fix:**

* Each edge region gets an SNS topic for its alarms. Alarm actions must be in
  the alarm's own region. Set `alarm_emails` in Terraform to subscribe
  addresses; each must confirm AWS's subscription email.
* Both alarms send `alarm_actions` and `ok_actions` to that topic.
* A new `degraded-rate` alarm fires above `degraded_rate_alarm` (default 5%)
  for 15 minutes. Its description points at
  `cloud_unavailable_total{reason}` for the cause.
* The topic is deliberately **not** encrypted with the AWS-managed `aws/sns`
  key: its key policy doesn't let CloudWatch publish, so notifications would be
  dropped silently. Use a customer-managed key if you need encryption.
* New static tests guard what `tofu validate` can't check: every alarm has
  actions, no alarm topic uses `aws/sns`, and the shutdown timings from
  question 19 stay in order.

## 44. Are record batches sized in the unit Firehose enforces?

**No.** Firehose limits a `PutRecordBatch` call to 4 MiB of *bytes*, but the
sink sized batches (and its queue budget from question 34) in *characters*.
CJK text is about 3 bytes per character, so a batch the sink thought was under
4 MB could be ~9 MB. Firehose rejected the whole call, and every record in it
was lost. That data comes from exactly the non-English traffic you most need
to calibrate on.

**Fix:** each record is encoded to UTF-8 once, in `submit`, and every limit is
measured in bytes from then on: the per-record maximum, the batch size and the
queue budget. A test batches about 12 MB of CJK records through a fake
Firehose that rejects any oversized call.

## 45. How wrong is the token estimate outside English?

**About 4× too low for CJK.** `estimate_tokens` assumed about 4 characters per
token, which is right for English and code. Chinese, Japanese and Korean run
close to one token per character. A long CJK prompt was estimated at a quarter
of its size, so the edge was asked for more `max_tokens` than its context had
room for (question 27). vLLM rejected the request, and every such request
escalated at full price.

**Fix:** ASCII characters still count at about 4 per token, and every other
character counts as about one token. This slightly over-estimates
accented-Latin text, which is the safe direction. Counting uses
`str.encode("ascii", "ignore")`, which runs in C.

## 46. Do escalations pay full price for prefixes Claude has already seen?

**Yes.** No escalation used prompt caching. A tenant's long system prompt was
sent uncached on every escalation, and each turn of a multi-turn conversation
re-sent the whole history at the full input price. For `claude-opus-5-5`, a
cache read costs $0.20/MTok against $4/MTok uncached, about 20× less.

**Fix (`cloud_prompt_caching`, on by default):**

* **System prompt:** marked with an explicit breakpoint, since it's the prefix
  shared across a tenant's escalations. Below the model's minimum cacheable
  size it silently isn't cached, at no cost.
* **Multi-turn conversations:** also get the top-level automatic breakpoint,
  because the next turn re-sends this history.
* **Single-turn requests:** deliberately skip the automatic breakpoint. They
  end in a unique question, so caching the tail would pay the 1.25× write
  premium on bytes that are never read back.
* **Cost accounting:** `cache_read_input_tokens` and
  `cache_creation_input_tokens` are priced separately
  (`cloud_price_cache_read_per_mtok`, `cloud_price_cache_write_per_mtok`),
  since `input_tokens` excludes them, so v0.2's estimates would have been
  wrong once caching was on. Responses report `cloud_cache_read_tokens` and
  `cloud_cache_write_tokens`.

## 47. What happens to empty content?

**It turned into a late, confusing failure, and caching made one case worse.**
The Claude API rejects empty text. After question 46, an empty system prompt
(a client sending `{"role": "system", "content": ""}`, or `system_prompt`
configured empty) went out as an empty text block with a `cache_control`
marker. The API returned a 400, and every such escalation degraded. Empty or
whitespace-only user and assistant messages passed validation and failed the
same way, only once they reached Claude.

**Fix:**

* An empty system prompt is left out of the Claude request entirely, which is
  the faithful translation, with or without caching.
* A user or assistant message with empty content is a conversation problem: a
  422 naming the message, before any model runs. An empty system message is
  still allowed.

Regression tests fail on the previous code.

## 48. What happens to judge labels if the run dies?

**They were all lost.** `hybrid-edge-label` kept every verdict in memory and
wrote the output file only after the last record was judged. Each verdict is a
billed Claude call, so a crash, a rate-limit storm or a Ctrl-C after thousands
of calls threw away everything already paid for.

**Fix:**

* The output is written atomically (temp file, then rename) every
  `--checkpoint-every` labels (default 100).
* It's written again in a `finally` on success, error or interrupt.
* Re-running with the output file as input resumes: labelled records are
  skipped, so only the remainder is billed. A test interrupts a run partway and
  checks that every finished label survives and the resume re-judges nothing.

## 49. Does a bad API key look like a Claude outage?

**Yes.** An empty or wrong Anthropic key secret makes every escalation fail
with 401/403, and those were counted as `reason="error"`, the same as Claude
being down. On-call would wait out an "outage" that is really a config mistake.

**Fix:** `AuthenticationError` and `PermissionDeniedError` get their own
reason, `auth`, in `cloud_unavailable_total`. They still trip the breaker,
since retrying a bad key won't fix it.

## 50. Does the image that ships actually start?

**Unverified.** CI built the Docker image but never ran it, so a missing
runtime dependency, a broken entrypoint or a failing `HEALTHCHECK` would have
shipped green.

**Fix:** CI now starts the built image with the mock edge, then:

* waits for Docker's own `HEALTHCHECK` to report `healthy`;
* runs one `/v1/infer` request (expecting an edge answer) and one
  `/v1/infer/stream` request (expecting a `done` event);
* checks the container runs as the non-root user (uid 10001);
* dumps the container logs if any step fails.

## 51. Can an edge report ready while it can't serve?

**Yes.** `/readyz` only checked that vLLM's `/models` endpoint returned 200. If
`edge_model` named a model the server isn't serving (a typo'd id, a
half-finished model swap), every generation failed with "model not found" and
escalated at full price. Readiness stayed green, so the ALB and Global
Accelerator kept routing that region's traffic to the broken edge instead of
failing over.

**Fix:** readiness also requires the configured model id to appear in
`/models`. A server that's reachable but has the wrong model loaded (or
nothing loaded yet) is now not ready, so its region is drained.

## 52. Which settings can silently send everything to Claude?

`edge_top_logprobs` was one. OpenAI-compatible servers cap `top_logprobs` at 20
(vLLM's `--max-logprobs`, set to 20 in Terraform). Any larger value turned every
edge call into a 400, and so every request into an escalation.

**Fix:** the setting is validated to the range 0–20 at startup, so a bad value
fails the deploy (and ECS's circuit breaker rolls it back) instead of
quietly burning budget.

## 53. Can a client fill gateway memory with one request?

**Yes.** `max_request_chars` (question 13) is checked on the *parsed*
messages, but by then FastAPI has already buffered and JSON-decoded the whole
body. Neither uvicorn nor the ALB caps body size, so one multi-gigabyte POST,
or an endless chunked upload, was read into memory before any limit applied.

**Fix:** `BodySizeLimit`, a raw ASGI middleware that applies before any parsing:

* A `Content-Length` over the cap gets a 413 at once. A malformed one gets a
  400.
* Chunked bodies, and bodies longer than their declared length, are counted
  as they arrive and cut off at the cap.
* On overflow the middleware sends the 413 itself, tells the app the client
  disconnected, and drops anything the app tries to send afterwards. The
  first version raised an exception instead, and a live test against uvicorn
  showed FastAPI's body parser catching it and answering 400. The unit tests
  had missed that.
* The cap defaults to `6 × max_request_chars + 64 KiB`. Six bytes per
  character covers clients that send ASCII-only JSON (`\uXXXX` escapes), so a
  request that passes the character limit is never cut off by the byte limit.
  `max_body_bytes` overrides it.

## 54. Is client identity safe by default?

**Only behind the ALB.** `trust_forwarded_for` defaulted to `true`, which is
correct behind the AWS ALB (it appends the real client address). But it was
wrong for every other way of running the gateway: docker-compose, a laptop, or
another platform without such a proxy. There, any client could forge
`X-Forwarded-For` and pick a new identity per request. That reopened the
per-caller budget bypass that question 32 closed.

**Fix:** the default is now `false`, because the gateway can't know whether a
trusted proxy sits in front of it. The Terraform deployment, which does know,
sets `HEI_TRUST_FORWARDED_FOR=true`. A static test fails if that line is ever
removed, since without it every caller behind the ALB would share the ALB's
address and one budget.

## 55. Can a calibration actually reach production?

**No. The flywheel led nowhere.** Shadow calls, the judge and the calibrator
produced `calibration.json`, but the gateway could only read a local path, the
Docker image doesn't include one, and Terraform had no setting for it. On AWS,
the only route was hand-building a custom image per calibration, so deployed
edges ran the uncalibrated heuristic indefinitely.

**Fix:**

* `gate_calibration_path` also accepts `s3://bucket/key`.
* Terraform's `gate_calibration_s3_uri` passes the URI to every edge, and
  grants `s3:GetObject` on **exactly that object**, not the whole bucket.
* Rolling out a calibration is changing the URI. Use a new key per version
  (`calibration-v3.json`): changing the URI rolls the edges, and reverting it
  rolls back.
* Startup safety from question 21 still applies. A calibration fit on a
  different edge model, or one that can't be read, stops the gateway from
  starting, and ECS's deployment circuit breaker rolls the change back rather
  than serving with it.

## 56. Does "deploy succeeded" mean the new version is running?

**No. `scripts/deploy.sh` had never been tested, and it had three flaws:**

* **Blind replication wait.** It waited a fixed `sleep 60` for cross-region
  ECR replication. A region the image hadn't reached yet failed to pull, and
  ECS rolled it back while the script carried on.
* **Unreviewed infrastructure changes.** Its `terraform apply -auto-approve`
  applied *every* pending infrastructure change along with the image tag, so
  an unrelated `tfvars` edit could ship unreviewed under the name "deploy".
* **No rollout check.** It exited once `apply` returned. A deployment the
  circuit breaker rolled back is "stable" too, on the old image, so it looked
  identical to success.

**Fix:**

* The script polls each edge region's ECR until the tag exists, and fails
  before applying anything if replication times out.
* `apply` is interactive by default and shows the full plan. `--yes` is the
  explicit opt-out for CI.
* After applying, it waits for each service to stabilise, then checks the
  *running* task definition's image tag. Any region still on the old image
  fails the deploy, by name.
* The script now has tests: they run it against stub `terraform`, `aws`,
  `docker` and `git` commands covering the happy path, slow replication, a
  replication timeout, interactive versus `--yes` apply, and a rolled-back
  region.

## 57. What can a compromised container reach?

**Too much.** Three gaps in the infrastructure:

* **Node credentials through IMDS.** With `awsvpc` tasks and an IMDS hop limit
  of 2, any process in the vLLM or gateway container could query instance
  metadata and take the **GPU node's** EC2 role (ECS agent and SSM
  permissions, broader than the task role). A model-server bug or a malicious
  dependency would get more than the task was ever granted.
* **Always-on ECS Exec.** `enable_execute_command = true` was hard-coded. Anyone
  with `ecs:ExecuteCommand` could open a shell next to the Anthropic key. It
  also didn't work: the task role lacked the `ssmmessages` permissions Exec
  needs.
* **Plain-HTTP bucket access.** The records buckets, which hold prompts and
  answers, accepted requests without TLS.

**Fix:**

* `ECS_AWSVPC_BLOCK_IMDS=true` on the container instances, IMDS hop limit 1,
  and IMDSv2 required. Tasks only have their own task role.
* ECS Exec is opt-in (`enable_execute_command`, default `false`). Turning it on
  also grants the task role the SSM channel permissions, so it actually works.
* Every records bucket, central or regional, has a policy denying any request
  where `aws:SecureTransport` is `false`.
* Static tests guard all three.

## 58. Can you deploy at peak load?

**No.** Every task needs a whole GPU (one per g6.xlarge). ECS rolling deploys
keep 100% of tasks healthy: they start each new task *before* stopping an old
one, so a deploy needs one spare GPU node. But the ASG's `max_size` equalled
`max_nodes`, which is also the service's autoscaling ceiling. At full scale,
exactly when a fix tends to be urgent, the new task couldn't be placed, the
deployment stalled, and the circuit breaker rolled it back.

**Fix:** the ASG allows `max_nodes + 1`. Tasks stay capped at `max_nodes`, so
the extra node exists only to make room during a deploy. Capacity-provider
managed scaling launches it when a task is waiting to be placed, and scales it
back in afterwards. It costs nothing at rest.

Also: Global Accelerator now health-checks every 10 s, not the default 30 s.
With three failures needed, a dead edge is drained in about 30 s instead of
about 90 s.

## 59. Can a region scale to zero when it's not needed?

**No, and the README said it could.** It suggested `min_nodes = 0` for
"burst-only" regions, but that does nothing:

* The service's autoscaling floor is `desired_tasks` (at least 1). ECS keeps
  that task running, and the capacity provider launches a GPU node for it
  regardless of `min_nodes`.
* Even at zero tasks, the region would never come back. With no healthy
  targets it receives no traffic, so the request-count scaling metric can't
  rise.

The real cost floor is one GPU node plus one NAT gateway per region, around
the clock.

**Fix:**

* The README now states that floor, and says the only way to stop paying for
  a region is to remove it from `edge_regions`. Global Accelerator then routes
  its users to the nearest remaining edge.
* The scaling settings are now checked when Terraform plans:
  `1 ≤ desired_tasks ≤ max_nodes` and `0 ≤ min_nodes ≤ max_nodes`.
  Inconsistent values used to fail deep inside an apply, or not fail at all.
* A test fails if the README ever promises `min_nodes = 0` again.

True scale-to-zero would need something to wake a region before traffic
arrives (scheduled scaling, or a Global Accelerator traffic dial driven by
demand elsewhere). That's added to "not changed".

## Not changed (and why)

* **Shared cache across nodes (Redis/ElastiCache):** it adds a dependency and
  network hops to every request. The per-node cache plus single-flight covers
  the hot path. Revisit if the hit rate matters at scale.
* **Global (cross-node) budget:** this needs shared state (DynamoDB or Redis).
  Per-node and per-key buckets cap the damage, and the escalation-rate alarm
  catches drift.
* **Scale-to-zero regions:** a region with no task gets no traffic, so it can
  never scale itself back up. It would need an external trigger (scheduled
  scaling, or demand-driven Global Accelerator traffic dials). Until then,
  each region's floor is one GPU node.
