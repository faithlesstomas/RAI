"""Optional NCSI sidecar commands."""

import os
from typing import Optional
import typer
from rai.core import console

neural = typer.Typer(help="NCSI sidecar operations.", no_args_is_help=True)


@neural.command(name="serve")
def neural_serve(  # noqa: PLR0913
    model_id: str = typer.Option(
        ..., "--model", help="Hugging Face model repository ID."
    ),
    model_revision: str = typer.Option(
        ..., "--revision", help="Immutable model commit/revision."
    ),
    tokenizer_revision: Optional[str] = typer.Option(
        None, "--tokenizer-revision", help="Tokenizer revision; defaults to --revision."
    ),
    lens_dir: Optional[str] = typer.Option(None, "--lens-dir", file_okay=False),
    device: str = typer.Option(
        "auto", "--device", help="Transformers device map (for example auto or cpu)."
    ),
    dtype: str = typer.Option(
        "auto", "--dtype", help="Torch dtype (for example auto, float16, bfloat16)."
    ),
    quantization: Optional[str] = typer.Option(
        None, "--quantization", help="Declared quantization identity for lens checks."
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Host used when --uds is omitted."
    ),
    port: int = typer.Option(
        8010, "--port", help="TCP port used when --uds is omitted."
    ),
    uds: Optional[str] = typer.Option(None, "--uds", dir_okay=False),
    max_concurrency: int = typer.Option(1, "--max-concurrency", min=1, max=32),
) -> None:
    """Run the isolated, read-only gcas.ncsi.v1 neural process."""
    from pathlib import Path  # noqa: PLC0415
    import uvicorn  # noqa: PLC0415
    from rai.neural.artifacts import LensRegistry  # noqa: PLC0415
    from rai.neural.engines.transformers import TransformersEngine  # noqa: PLC0415
    from rai.neural.server import create_neural_app  # noqa: PLC0415
    from rai.paths import data_dir  # noqa: PLC0415

    registry = LensRegistry(
        Path(lens_dir) if lens_dir else data_dir() / "neural" / "lenses"
    )
    engine = TransformersEngine(
        model_id,
        model_revision,
        tokenizer_revision,
        registry,
        device=device,
        dtype=dtype,
        quantization=quantization,
    )
    app = create_neural_app(engine, registry, max_concurrency=max_concurrency)
    if uds:
        socket_path = Path(uds)
        socket_path.parent.mkdir(mode=448, parents=True, exist_ok=True)
        console.print(
            f"Starting NCSI neural sidecar on Unix socket: [bold green]{socket_path}[/bold green]"
        )
        previous_umask = os.umask(63)
        try:
            uvicorn.run(app, uds=str(socket_path), log_config=None)
        finally:
            os.umask(previous_umask)
    else:
        console.print(
            f"Starting NCSI neural sidecar on [bold green]http://{host}:{port}[/bold green]"
        )
        uvicorn.run(app, host=host, port=port, log_config=None)


@neural.command(name="fit-lens")
def neural_fit_lens(  # noqa: PLR0913
    model_id: str = typer.Option(..., "--model"),
    model_revision: str = typer.Option(..., "--revision"),
    tokenizer_revision: Optional[str] = typer.Option(None, "--tokenizer-revision"),
    lens_id: str = typer.Option(..., "--lens-id"),
    lens_revision: str = typer.Option(..., "--lens-revision"),
    prompts_path: str = typer.Option(..., "--prompts", exists=True),
    output_dir: str = typer.Option(..., "--output"),
    corpus_id: str = typer.Option(..., "--corpus-id"),
    corpus_license: str = typer.Option(..., "--corpus-license"),
    layers: list[int] = typer.Option(None, "--layer", min=0),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    quantization: Optional[str] = typer.Option(None, "--quantization"),
    dim_batch: int = typer.Option(8, "--dim-batch", min=1),
    max_seq_len: int = typer.Option(128, "--max-seq-len", min=2),
    skip_first: int = typer.Option(16, "--skip-first", min=0),
) -> None:
    """Fit the pinned reference J-lens and write a safe versioned artifact."""
    from pathlib import Path  # noqa: PLC0415
    from rai.neural.fitting import FitConfig, fit_lens_artifact  # noqa: PLC0415

    manifest = fit_lens_artifact(
        FitConfig(
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision or model_revision,
            lens_id=lens_id,
            lens_revision=lens_revision,
            prompts_path=Path(prompts_path),
            output_dir=Path(output_dir),
            calibration_corpus=corpus_id,
            calibration_license=corpus_license,
            layers=tuple(layers or ()),
            device=device,
            dtype=dtype,
            quantization=quantization,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
        )
    )
    console.print(
        f"Fitted [bold]{manifest.lens_id}@{manifest.lens_revision}[/bold] for layers {list(manifest.layers)}"
    )
