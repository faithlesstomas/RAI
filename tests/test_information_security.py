"""Security regression tests for compatibility information tools."""

import pytest

from rai.tools.information import calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("round(sin(pi / 2), 2)", "1.0"),
        ("sum([1, 2, 3])", "6"),
        ("max(2, abs(-5))", "5"),
    ],
)
def test_calculator_supports_bounded_arithmetic(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "().__class__.__base__.__subclasses__()",
        "__import__('os')",
        "(lambda: 1)()",
        "open('/etc/passwd').read()",
        "2 ** 101",
        "[1] * 1000",
    ],
)
def test_calculator_rejects_python_and_unbounded_operations(expression: str) -> None:
    assert calculate(expression).startswith("Error:")


def test_calculator_limits_input_size_and_complexity() -> None:
    assert calculate("1" * 257).startswith("Error:")
    assert calculate("+".join("1" for _ in range(40))).startswith("Error:")
