from types import SimpleNamespace

import pytest

from research_pilot.config import load_settings


@pytest.fixture
def settings(tmp_path):
    return load_settings(
        provider="anthropic",
        model="claude-opus-5-5",
        max_steps=3,
        defenses=False,  # tests opt in to week 3 defenses explicitly
        tavily_api_key="tvly-test",
        cache_dir=tmp_path / "cache",
        trace_dir=tmp_path / "traces",
    )


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, tool_input, block_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def response(stop_reason, *blocks):
    usage = SimpleNamespace(input_tokens=100, output_tokens=20)
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks), usage=usage)


class FakeClient:
    """Stands in for anthropic.Anthropic: returns scripted responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)
