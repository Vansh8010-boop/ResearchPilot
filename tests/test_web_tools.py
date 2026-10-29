import json

import httpx

from research_pilot.tools import Toolbox
from research_pilot.tools.fetch import MAX_CHARS

ARTICLE = "<html><body><article><h1>Mars</h1>" + "<p>Mars is the fourth planet from the Sun.</p>" * 20 + "</article></body></html>"


def make_toolbox(settings, handler):
    calls = []

    def recording_handler(request):
        calls.append(request)
        return handler(request)

    return Toolbox(settings, http=httpx.Client(transport=httpx.MockTransport(recording_handler))), calls


def test_search_returns_trimmed_results(settings):
    def handler(request):
        assert request.headers["Authorization"] == "Bearer tvly-test"
        assert json.loads(request.content)["max_results"] == 2
        return httpx.Response(200, json={"results": [
            {"title": "Mars", "url": "https://example.com/mars", "content": "Fourth planet", "score": 0.9},
        ]})

    toolbox, _ = make_toolbox(settings, handler)
    result = toolbox.run("web_search", {"query": "mars", "k": 2})
    assert result.ok
    assert json.loads(result.content) == [{"title": "Mars", "url": "https://example.com/mars", "snippet": "Fourth planet"}]


def test_search_http_error_is_reported(settings):
    toolbox, _ = make_toolbox(settings, lambda r: httpx.Response(429, text="rate limited"))
    result = toolbox.run("web_search", {"query": "mars"})
    assert not result.ok and "429" in result.content


def test_search_results_are_cached(settings):
    toolbox, calls = make_toolbox(settings, lambda r: httpx.Response(200, json={"results": [{"title": "t", "url": "u", "content": "c"}]}))
    first = toolbox.run("web_search", {"query": "mars"})
    second = toolbox.run("web_search", {"query": "mars"})
    assert len(calls) == 1
    assert second.cached and second.content == first.content


def test_fetch_extracts_main_text(settings):
    toolbox, _ = make_toolbox(settings, lambda r: httpx.Response(200, html=ARTICLE))
    result = toolbox.run("fetch_page", {"url": "https://example.com/mars"})
    assert result.ok
    assert "fourth planet" in result.content
    assert "<p>" not in result.content


def test_fetch_truncates_long_pages(settings):
    long_page = "<html><body><article>" + "<p>" + "word " * 5000 + "</p></article></body></html>"
    toolbox, _ = make_toolbox(settings, lambda r: httpx.Response(200, html=long_page))
    result = toolbox.run("fetch_page", {"url": "https://example.com/long"})
    assert result.ok
    assert "[truncated:" in result.content
    assert len(result.content) < MAX_CHARS + 100


def test_fetch_errors(settings):
    toolbox, _ = make_toolbox(settings, lambda r: httpx.Response(404))
    assert "404" in toolbox.run("fetch_page", {"url": "https://example.com/missing"}).content

    toolbox, _ = make_toolbox(settings, lambda r: httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}))
    assert "unsupported content type" in toolbox.run("fetch_page", {"url": "https://example.com/a.pdf"}).content

    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    toolbox, _ = make_toolbox(settings, timeout)
    assert "timed out" in toolbox.run("fetch_page", {"url": "https://example.com/slow"}).content

    assert not toolbox.run("fetch_page", {"url": "file:///etc/passwd"}).ok


def test_toolbox_validates_input(settings):
    toolbox, _ = make_toolbox(settings, lambda r: httpx.Response(500))
    assert "unknown tool" in toolbox.run("rm_rf", {}).content
    assert "missing required" in toolbox.run("web_search", {}).content
