import json

import pytest

requests = pytest.importorskip("requests")

from papertrail.llm import base, groq, hf, ollama  # noqa: E402


class _Resp:
    def __init__(self, status=200, lines=()):
        self.status_code = status
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


def _sse(*tokens):
    out = [("data: " + json.dumps({"choices": [{"delta": {"content": t}}]})).encode() for t in tokens]
    return out + [b"data: [DONE]"]


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setattr(hf, "HF_TOKEN", "hf_test")
    monkeypatch.setattr(groq, "GROQ_API_KEY", "gsk_test")


def test_hf_streams_tokens(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, _sse("Hello", " world")))
    s = hf.generate_answer_hf_stream("q", "ev")
    assert s.collect() == "Hello world"
    assert s.status == base.OK


def test_hf_sends_chat_payload(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen.update(url=url, **kw)
        return _Resp(200, _sse("x"))

    monkeypatch.setattr(requests, "post", fake)
    hf.generate_answer_hf_stream("q?", "ev").collect()
    assert seen["url"].startswith("https://router.huggingface.co/")
    assert seen["json"]["messages"][1]["content"].endswith("QUESTION:\nq?")
    assert seen["headers"]["Authorization"] == "Bearer hf_test"


def test_hf_no_token(monkeypatch):
    monkeypatch.setattr(hf, "HF_TOKEN", None)
    s = hf.generate_answer_hf_stream("q", "ev")
    assert s.collect() == "" and s.status == base.NO_TOKEN


@pytest.mark.parametrize("code,expected", [
    (401, base.AUTH), (403, base.AUTH), (402, base.RATE_LIMITED), (429, base.RATE_LIMITED),
    (404, base.UNAVAILABLE), (410, base.UNAVAILABLE), (500, base.ERROR),
])
def test_hf_http_errors_give_reason(monkeypatch, code, expected):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(code))
    s = hf.generate_answer_hf_stream("q", "ev")
    assert s.collect() == "" and s.status == expected


def test_hf_timeout(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(requests, "post", boom)
    s = hf.generate_answer_hf_stream("q", "ev")
    assert s.collect() == "" and s.status == base.TIMEOUT


def test_empty_stream_reports_empty(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, [b"data: [DONE]"]))
    s = hf.generate_answer_hf_stream("q", "ev")
    assert s.collect() == "" and s.status == base.EMPTY


def test_ollama_not_reachable(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(requests, "post", boom)
    s = ollama.generate_answer_ollama_stream("q", "ev")
    assert s.collect() == "" and s.status == base.NOT_REACHABLE


def test_origin_includes_reason():
    from papertrail import config, qa
    tag = qa.answer_origin(config.MODE_HF, False, False, base.NO_TOKEN)
    assert tag == f"{qa.ORIGIN_UNAVAILABLE} ({base.NO_TOKEN})"


def test_groq_uses_groq_endpoint_and_key(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen.update(url=url, **kw)
        return _Resp(200, _sse("Hi"))

    monkeypatch.setattr(requests, "post", fake)
    s = groq.generate_answer_groq_stream("q", "ev")
    assert s.collect() == "Hi" and s.status == base.OK
    assert seen["url"].startswith("https://api.groq.com/openai/v1/")
    assert seen["headers"]["Authorization"] == "Bearer gsk_test"
    assert seen["json"]["model"]


def test_groq_no_key(monkeypatch):
    monkeypatch.setattr(groq, "GROQ_API_KEY", None)
    s = groq.generate_answer_groq_stream("q", "ev")
    assert s.collect() == "" and s.status == base.NO_TOKEN


def test_groq_mode_dispatch(monkeypatch):
    from papertrail import config, qa
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, _sse("ok")))
    assert qa._generate(config.MODE_GROQ, "q", "ev") == ("ok", base.OK)
    assert qa._generate(config.MODE_QUOTES, "q", "ev") == ("", base.OK)
