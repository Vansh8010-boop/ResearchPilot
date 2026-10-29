import pytest

from research_pilot.tools.calculator import calculator


@pytest.mark.parametrize("expr, expected", [
    ("2 + 3 * 4", "14"),
    ("(2 + 3) * 4", "20"),
    ("7 / 2", "3.5"),
    ("2 ** 10", "1024"),
    ("-5 + 3", "-2"),
    ("sqrt(16)", "4"),
    ("round(pi, 2)", "3.14"),
    ("125.7e6 / 695662", "180.69"),
    ("max(3, 9, 4)", "9"),
])
def test_valid_expressions(expr, expected):
    result = calculator(expr)
    assert result.ok, result.content
    assert result.content.startswith(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('secrets.txt')",
    "(1).__class__",
    "[1, 2, 3]",
    "x + 1",
    "lambda: 1",
])
def test_rejects_non_arithmetic(expr):
    assert not calculator(expr).ok


def test_division_by_zero():
    result = calculator("1 / 0")
    assert not result.ok and "division by zero" in result.content


def test_huge_exponent_blocked():
    assert not calculator("9 ** 99999999").ok


def test_commas_get_helpful_error():
    result = calculator("1,234,567 / 2")
    assert not result.ok and "commas" in result.content
