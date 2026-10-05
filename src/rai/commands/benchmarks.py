"""Bounded assistant benchmark runners."""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
import click
from .settings import _assistant_config


def _run_assistant_benchmark(  # noqa: PLR0913
    backend: str,
    model: str | None,
    profile: str | None,
    corpus: Path,
    output: Path,
    energy: str,
    require_energy: bool,
    max_input_tokens: int,
    max_output_tokens: int,
    max_latency_seconds: float,
    trials: int,
) -> None:
    """Run the isolated equal-budget memory benchmark and persist its manifest."""
    from returns.result import Success  # noqa: PLC0415

    from rai.assistant.benchmark import run_retrieval_benchmark  # noqa: PLC0415
    from rai.assistant.energy import LinuxEnergyMeter  # noqa: PLC0415
    from rai.assistant.runtime import (  # noqa: PLC0415
        build_assistant_backend,
        resolve_assistant_config,
    )

    try:
        config = _assistant_config(backend, model, profile)
        assistant_config = dict(config.get("assistant", {}))
        assistant_config["max_output_tokens"] = max_output_tokens
        config["assistant"] = assistant_config
        runtime = resolve_assistant_config(config)
        model_backend = build_assistant_backend(runtime)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    meter = LinuxEnergyMeter() if energy == "auto" else None

    async def _run() -> None:
        result = await run_retrieval_benchmark(
            model_backend,
            corpus_path=corpus,
            output_path=output,
            energy_meter=meter,
            require_energy=require_energy,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_latency_seconds=max_latency_seconds,
            trials=trials,
        )
        if not isinstance(result, Success):
            failure = result.failure()
            raise click.ClickException(f"[{failure.code}] {failure.message}")
        artifact = result.unwrap()
        click.echo(
            json.dumps(
                {
                    "output": str(output),
                    "backend": artifact.backend_name,
                    "model": artifact.model_name,
                    "corpus": artifact.run.corpus_version,
                    "channels": [
                        aggregate.model_dump(mode="json")
                        for aggregate in artifact.aggregates
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    asyncio.run(_run())


def _run_assistant_dialog_benchmark(  # noqa: PLR0913
    backend: str,
    model: str | None,
    profile: str | None,
    corpus: Path,
    output: Path,
    max_output_tokens: int,
) -> None:
    """Run the multi-turn conversational benchmark and persist its report."""
    from returns.result import Success  # noqa: PLC0415

    from rai.assistant.conversational_evaluation import (  # noqa: PLC0415
        run_conversational_benchmark,
    )
    from rai.assistant.runtime import (  # noqa: PLC0415
        build_assistant_backend,
        resolve_assistant_config,
    )

    try:
        config = _assistant_config(backend, model, profile)
        assistant_config = dict(config.get("assistant", {}))
        assistant_config["max_output_tokens"] = max_output_tokens
        config["assistant"] = assistant_config
        runtime = resolve_assistant_config(config)
        model_backend = build_assistant_backend(runtime)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    async def _run() -> None:
        result = await run_conversational_benchmark(
            model_backend,
            corpus_path=corpus,
            output_path=output,
        )
        if not isinstance(result, Success):
            failure = result.failure()
            raise click.ClickException(f"[{failure.code}] {failure.message}")
        report = result.unwrap()
        click.echo(
            json.dumps(
                {
                    "output": str(output),
                    "backend": report.backend_name,
                    "model": report.model_name,
                    "corpus": report.corpus_version,
                    "scenarios": report.scenario_count,
                    "total_turns": report.total_turns,
                    "mean_coherence": report.mean_coherence,
                    "parroting_rate": report.parroting_rate,
                    "repetition_rate": report.repetition_rate,
                    "role_confusion_rate": report.role_confusion_rate,
                    "mean_latency_ms": report.mean_latency_ms,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    asyncio.run(_run())


def _run_assistant_context_rot_benchmark(  # noqa: PLR0913
    backend: str,
    model: str | None,
    profile: str | None,
    token_steps_str: str,
    depths_str: str,
    trials: int,
    corpus: Path,
    output: Path,
    server_context_window: int | None,
    max_output_tokens: int,
    max_latency_seconds: float,
) -> None:
    """Run the context rot A/B benchmark and format summary output."""
    from returns.result import Success  # noqa: PLC0415

    from rai.assistant.context_rot_evaluation import (  # noqa: PLC0415
        evaluate_context_rot,
        format_context_rot_summary_table,
        load_context_rot_corpus,
        save_context_rot_report,
    )
    from rai.assistant.runtime import (  # noqa: PLC0415
        build_assistant_backend,
        resolve_assistant_config,
    )

    try:
        config = _assistant_config(backend, model, profile)
        assistant_config = dict(config.get("assistant", {}))
        assistant_config["max_output_tokens"] = max_output_tokens
        config["assistant"] = assistant_config
        runtime = resolve_assistant_config(config)
        model_backend = build_assistant_backend(runtime)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        corpus_data = load_context_rot_corpus(corpus)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"Failed to load context rot corpus: {exc}") from exc

    try:
        token_steps = tuple(
            int(s.strip()) for s in token_steps_str.split(",") if s.strip()
        )
    except ValueError as exc:
        raise click.ClickException(
            f"Invalid token-steps format: {token_steps_str}"
        ) from exc

    depths = tuple(d.strip() for d in depths_str.split(",") if d.strip())
    for d in depths:
        if d not in ("start", "middle", "end"):
            raise click.ClickException(
                f"Invalid depth: {d}. Expected start, middle, or end."
            )

    async def _run() -> None:
        result = await evaluate_context_rot(
            model_backend,
            corpus=corpus_data,
            token_steps=token_steps,
            depths=depths,  # type: ignore[arg-type]
            trials=trials,
            max_output_tokens=max_output_tokens,
            max_latency_seconds=max_latency_seconds,
            configured_context_window=(
                server_context_window
                if server_context_window is not None
                else runtime.context_window
                if runtime.backend == "llama"
                else None
            ),
        )
        if not isinstance(result, Success):
            failure = result.failure()
            raise click.ClickException(f"[{failure.code}] {failure.message}")
        report = result.unwrap()
        save_context_rot_report(report, output)
        click.echo(format_context_rot_summary_table(report))
        click.echo(
            json.dumps(
                {
                    "output": str(output),
                    "backend": report.backend_name,
                    "model": report.model_name,
                    "token_steps": list(report.token_steps),
                    "total_cases": report.total_cases,
                    "successful_cases": report.successful_cases,
                    "failed_cases": report.failed_cases,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    try:
        asyncio.run(_run())
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
