"""Design review questions 51-52: readiness and config must match what the edge can serve."""

import httpx
import pytest

from hybrid_edge.backends.openai_compat import OpenAICompatBackend
from hybrid_edge.config import Settings


def backend(model, handler):
    b = OpenAICompatBackend("http://edge/v1", model)
    b._client = httpx.AsyncClient(base_url="http://edge/v1", transport=httpx.MockTransport(handler))
    return b


def models(*ids, status=200):
    return lambda req: httpx.Response(status, json={"object": "list", "data": [{"id": i} for i in ids]})


@pytest.mark.parametrize("handler,ready", [
    (models("Qwen/Qwen2.5-7B-Instruct", "other"), True),
    (models("meta-llama/Llama-3.1-8B-Instruct"), False),   # wrong model loaded
    (models(), False),                                      # still loading / nothing served
    (models("Qwen/Qwen2.5-7B-Instruct", status=503), False),
    (lambda req: httpx.Response(200, text="not json"), False),
])
async def test_ready_requires_the_configured_model(handler, ready):
    b = backend("Qwen/Qwen2.5-7B-Instruct", handler)
    assert await b.ready() is ready
    await b.aclose()


async def test_ready_false_when_unreachable():
    def boom(req):
        raise httpx.ConnectError("refused", request=req)

    b = backend("m", boom)
    assert await b.ready() is False
    await b.aclose()


@pytest.mark.parametrize("value,ok", [(0, True), (5, True), (20, True), (21, False), (-1, False)])
def test_edge_top_logprobs_bounds(value, ok):
    if ok:
        assert Settings.from_dict({"edge_top_logprobs": value}).edge_top_logprobs == value
    else:
        with pytest.raises(ValueError, match="edge_top_logprobs"):
            Settings.from_dict({"edge_top_logprobs": value})
