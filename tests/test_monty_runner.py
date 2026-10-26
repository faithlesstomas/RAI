"""
Unit tests for MontyRunner and its persistent REPL snapshots.
"""
from __future__ import annotations

import pytest
from returns.result import Failure, Success

from rai.container import ApplicationContainer
from rai.tools.security.monty import MontyRunner, get_monty_runner


def test_get_monty_runner_factory() -> None:
    runner = get_monty_runner()
    assert isinstance(runner, MontyRunner)
    assert runner.is_available()


@pytest.mark.asyncio
async def test_container_monty_runner_managed_lifecycle() -> None:
    container = ApplicationContainer(config={}, testing=True)
    runner1 = container.monty_runner
    runner2 = container.monty_runner
    assert runner1 is runner2
    assert runner1.is_available()

    await container.close()
    assert container._monty_runner is None


def test_monty_runner_eval_pure_expression() -> None:
    runner = MontyRunner()
    res = runner.run("2 + 2")
    assert isinstance(res, Success)
    result = res.unwrap()
    assert result.output == "4"
    expected_return = 4
    assert result.return_value == expected_return
    assert result.duration_ms >= 0.0
    assert len(result.session_blob) > 0


def test_monty_runner_empty_code() -> None:
    runner = MontyRunner()
    res = runner.run("   ")
    assert isinstance(res, Failure)
    assert res.failure() == "EMPTY_CODE"


def test_monty_runner_syntax_error() -> None:
    runner = MontyRunner()
    res = runner.run("def invalid syntax")
    assert isinstance(res, Failure)
    assert "MONTY_EXECUTION_ERROR" in res.failure()


def test_monty_runner_persistent_session_state() -> None:
    runner = MontyRunner()
    # Turn 1: define variables and data
    res1 = runner.run("x = 40\ny = 2\nnumbers = [1, 2, 3]")
    assert isinstance(res1, Success)
    step1 = res1.unwrap()
    assert "x" in step1.variables
    assert step1.variables["x"] == "int"
    assert "numbers" in step1.variables
    assert step1.variables["numbers"] == "list"

    # Turn 2: use variables from previous turn with session_blob
    res2 = runner.run("x + y", session_blob=step1.session_blob)
    assert isinstance(res2, Success)
    step2 = res2.unwrap()
    assert step2.output == "42"


def test_monty_runner_persistent_function_definitions() -> None:
    runner = MontyRunner()
    code1 = """
def multiply(a, b):
    return a * b
"""
    res1 = runner.run(code1)
    assert isinstance(res1, Success)
    step1 = res1.unwrap()
    assert "multiply" in step1.variables
    assert step1.variables["multiply"] == "function"

    # Turn 2: invoke the function defined in turn 1
    code2 = "multiply(6, 7)"
    res2 = runner.run(code2, session_blob=step1.session_blob)
    assert isinstance(res2, Success)
    step2 = res2.unwrap()
    assert step2.output == "42"


def test_monty_runner_isolation_blocks_system_calls() -> None:
    runner = MontyRunner()
    # Monty does not have system() in os module
    res = runner.run("import os\nos.system('whoami')")
    assert isinstance(res, Failure)
    assert "MONTY_EXECUTION_ERROR" in res.failure()


def test_monty_runner_infinite_loop_timeout() -> None:
    runner = MontyRunner(max_duration_secs=0.2)
    # Loop should be terminated by duration quota
    res = runner.run("while True: pass")
    assert isinstance(res, Failure)
    assert "MONTY_EXECUTION_ERROR" in res.failure()


def test_monty_runner_inspect_state() -> None:
    runner = MontyRunner()
    res1 = runner.run("a = 42\nb = 'hello'")
    assert isinstance(res1, Success)
    blob = res1.unwrap().session_blob
    assert blob is not None

    inspect_res = runner.inspect_state(blob)
    assert isinstance(inspect_res, Success)
    vars_dict = inspect_res.unwrap()
    assert vars_dict.get("a") == "int"
    assert vars_dict.get("b") == "str"


def test_monty_runner_inspect_values() -> None:
    runner = MontyRunner()
    res1 = runner.run("a = 42\nitems = [1, 2, 3]")
    assert isinstance(res1, Success)
    blob = res1.unwrap().session_blob
    assert blob is not None

    values_res = runner.inspect_values(blob)
    assert isinstance(values_res, Success)
    details = values_res.unwrap()
    assert details["a"] == {"type": "int", "value": "42"}
    assert details["items"] == {"type": "list", "value": "[1, 2, 3]"}
