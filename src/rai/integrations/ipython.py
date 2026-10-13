"""
IPython and Jupyter magic extension for Rich AI (RAI).

Usage in Jupyter / IPython:
    %load_ext rai.integrations.ipython
    %rai ask How do I compute a Fourier transform in Python?
    %%rai
    Explain the function below and write unit tests for it:
    def add(a, b): return a + b
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

try:
    from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
except ImportError:  # pragma: no cover
    # Fallback dummy class if IPython is not installed
    class Magics:  # type: ignore[no-redef]
        def __init__(self, shell: object = None) -> None:
            self.shell = shell

    def magics_class(cls: type) -> type:  # type: ignore[no-redef]
        return cls

    def line_magic(func: object) -> object:  # type: ignore[no-redef]
        return func

    def cell_magic(func: object) -> object:  # type: ignore[no-redef]
        return func


@magics_class
class RAIMagics(Magics):
    """IPython Magics for interacting with RAI Assistant and persistent REPL."""

    def __init__(self, shell: object = None, session_id: Optional[str] = None) -> None:
        super().__init__(shell)
        self.session_id = session_id or "ipython-session"
        self._container: Any = None
        self._service: Any = None

    def _get_service(self) -> Any:  # noqa: ANN401
        if self._service is None:
            from rai.container import ApplicationContainer  # noqa: PLC0415
            from rai.commands.settings import _assistant_config  # noqa: PLC0415

            self._container = ApplicationContainer(
                config=_assistant_config(backend="auto", model=None)
            )
            self._service = self._container.assistant_service
        return self._service

    @line_magic
    def rai(self, line: str) -> None:
        """Execute a line command for RAI: %rai ask <prompt> or %rai repl."""
        clean = line.strip()
        if not clean or clean == "help":
            print(
                "RAI IPython Magic Usage:\n"
                "  %rai ask <prompt>     - Send a query to the RAI assistant\n"
                "  %rai repl             - Show active variables in Monty REPL\n"
                "  %rai session <id>     - Set current RAI session ID\n"
                "  %%rai                 - Multi-line cell prompt to assistant\n"
            )
            return

        if clean.startswith("session "):
            self.session_id = clean.removeprefix("session ").strip()
            print(f"RAI session switched to: {self.session_id}")
            return

        if clean == "repl":
            self._show_repl_state()
            return

        prompt = clean.removeprefix("ask ").strip() if clean.startswith("ask ") else clean
        self._ask_assistant(prompt)

    @cell_magic
    def rai_cell(self, line: str, cell: str) -> None:
        """Multi-line cell prompt to RAI Assistant: %%rai."""
        prompt = cell.strip()
        if line.strip():
            prompt = f"{line.strip()}\n\n{prompt}"
        self._ask_assistant(prompt)

    def _show_repl_state(self) -> None:
        from returns.result import Success  # noqa: PLC0415
        from rai.tools.security.monty import get_monty_runner  # noqa: PLC0415

        service = self._get_service()

        async def _fetch() -> None:
            saved = await service.store.get_repl_session(self.session_id)
            if isinstance(saved, Success) and saved.unwrap():
                blob, vars_dict = saved.unwrap()
                if not vars_dict and blob:
                    runner = get_monty_runner()
                    res = runner.inspect_state(blob)
                    if isinstance(res, Success):
                        vars_dict = res.unwrap()
                if vars_dict:
                    print(f"Active REPL variables ({self.session_id}):")
                    for k, v in sorted(vars_dict.items()):
                        print(f"  {k}: {v}")
                else:
                    print("REPL session has no active variables.")
            else:
                print("REPL session has no active variables.")

        asyncio.run(_fetch())

    def _ask_assistant(self, prompt: str) -> None:
        from returns.result import Success  # noqa: PLC0415
        from rai.assistant.records import ConversationTurn  # noqa: PLC0415
        from rai.kernel.records import DataClass, ProducerIdentity, _new_id  # noqa: PLC0415

        service = self._get_service()

        turn = ConversationTurn(
            record_id=_new_id(),
            producer=ProducerIdentity(
                producer_id="ipython-magic", kind="user", version="1.0.0"
            ),
            session_id=self.session_id,
            role="user",
            data_class=DataClass.LOCAL,
            text=prompt,
        )

        async def _run() -> None:
            start_res = await service.start()
            if not isinstance(start_res, Success):
                err = start_res.failure()
                print(f"Error starting assistant: [{err.code}] {err.message}")
                return
            res = await service.accept_turn(turn)
            if isinstance(res, Success):
                print(res.unwrap().text)
            else:
                err = res.failure()
                print(f"Error: [{err.code}] {err.message}")

        asyncio.run(_run())


def load_ipython_extension(ipython: Any) -> None:  # noqa: ANN401
    """Entry point for IPython `%load_ext rai.integrations.ipython`."""
    ipython.register_magics(RAIMagics)
