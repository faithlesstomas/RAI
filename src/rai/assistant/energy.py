"""Local, inspectable energy measurement for assistant evaluation runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Protocol

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure

from .records import make_assistant_failure

_MINIMUM_POWER_SAMPLES = 2


@dataclass(frozen=True)
class EnergyMeasurement:
    """Measured energy and enough provenance to interpret its scope."""

    joules: float
    source: str
    scope: str
    sensors: tuple[str, ...]
    sample_count: int
    duration_seconds: float


class EnergyMeasurementSession(Protocol):
    """One measurement interval around a bounded operation."""

    async def finish(self) -> Result[EnergyMeasurement, ActionFailure]: ...


class EnergyMeter(Protocol):
    """Starts an independently inspectable local energy interval."""

    async def start(self) -> Result[EnergyMeasurementSession, ActionFailure]: ...


def _read_number(path: Path) -> float:
    return float(path.read_text(encoding="ascii").strip())


@dataclass
class _RaplSession:
    readings: tuple[tuple[Path, Path, float], ...]
    started_at: float

    async def finish(self) -> Result[EnergyMeasurement, ActionFailure]:
        try:
            total_microjoules = 0.0
            names: list[str] = []
            for energy_path, maximum_path, initial in self.readings:
                final = _read_number(energy_path)
                maximum = _read_number(maximum_path)
                delta = final - initial if final >= initial else maximum - initial + final
                total_microjoules += max(0.0, delta)
                names.append(str(energy_path.parent))
            duration = max(0.0, time.perf_counter() - self.started_at)
            return Success(
                EnergyMeasurement(
                    joules=total_microjoules / 1_000_000.0,
                    source="linux-rapl",
                    scope="cpu-package",
                    sensors=tuple(names),
                    sample_count=2,
                    duration_seconds=duration,
                )
            )
        except (OSError, ValueError) as exc:
            return Failure(
                make_assistant_failure(
                    code="ENERGY_MEASUREMENT_FAILED",
                    message=f"failed to finish RAPL measurement: {exc}",
                )
            )


class _PowerSensorSession:
    def __init__(
        self,
        sensors: tuple[Path, ...],
        scopes: tuple[str, ...],
        interval_seconds: float,
    ) -> None:
        self.sensors = sensors
        self.scopes = scopes
        self.interval_seconds = interval_seconds
        self.started_at = time.perf_counter()
        self._samples: list[tuple[float, float]] = []
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._sample())

    def _read_watts(self) -> float:
        return sum(_read_number(path) for path in self.sensors) / 1_000_000.0

    async def _sample(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            try:
                self._samples.append((now, self._read_watts()))
            except (OSError, ValueError):
                return
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval_seconds)
            except TimeoutError:
                continue

    async def finish(self) -> Result[EnergyMeasurement, ActionFailure]:
        self._stop.set()
        await self._task
        try:
            self._samples.append((time.perf_counter(), self._read_watts()))
        except (OSError, ValueError) as exc:
            return Failure(
                make_assistant_failure(
                    code="ENERGY_MEASUREMENT_FAILED",
                    message=f"failed to finish hwmon measurement: {exc}",
                )
            )
        if len(self._samples) < _MINIMUM_POWER_SAMPLES:
            return Failure(
                make_assistant_failure(
                    code="ENERGY_MEASUREMENT_FAILED",
                    message="hwmon measurement returned fewer than two samples",
                )
            )
        joules = sum(
            (right_time - left_time) * (left_watts + right_watts) / 2.0
            for (left_time, left_watts), (right_time, right_watts) in zip(
                self._samples, self._samples[1:]
            )
        )
        return Success(
            EnergyMeasurement(
                joules=max(0.0, joules),
                source="linux-hwmon-power",
                scope="unattributed:" + ",".join(self.scopes),
                sensors=tuple(str(path) for path in self.sensors),
                sample_count=len(self._samples),
                duration_seconds=max(0.0, self._samples[-1][0] - self.started_at),
            )
        )


class LinuxEnergyMeter:
    """Prefer package RAPL counters, then integrate readable hwmon power sensors."""

    def __init__(
        self,
        *,
        powercap_root: Path = Path("/sys/class/powercap"),
        hwmon_root: Path = Path("/sys/class/hwmon"),
        sample_interval_seconds: float = 0.05,
    ) -> None:
        self.powercap_root = powercap_root
        self.hwmon_root = hwmon_root
        self.sample_interval_seconds = sample_interval_seconds

    def _rapl_readings(self) -> tuple[tuple[Path, Path, float], ...]:
        readings: list[tuple[Path, Path, float]] = []
        for directory in sorted(self.powercap_root.glob("intel-rapl:*")):
            if directory.name.count(":") != 1:
                continue
            energy = directory / "energy_uj"
            maximum = directory / "max_energy_range_uj"
            try:
                readings.append((energy, maximum, _read_number(energy)))
            except (OSError, ValueError):
                continue
        return tuple(readings)

    def _power_sensors(self) -> tuple[tuple[Path, str], ...]:
        sensors: list[tuple[Path, str]] = []
        for directory in sorted(self.hwmon_root.glob("hwmon*")):
            name_path = directory / "name"
            try:
                name = name_path.read_text(encoding="ascii").strip().casefold()
            except OSError:
                continue
            if name.startswith("bat"):
                continue
            candidates = tuple(directory.glob("power*_average")) or tuple(
                directory.glob("power*_input")
            )
            for candidate in candidates:
                try:
                    _read_number(candidate)
                except (OSError, ValueError):
                    continue
                label_path = candidate.with_name(
                    candidate.name.replace("_average", "_label").replace(
                        "_input", "_label"
                    )
                )
                try:
                    label = label_path.read_text(encoding="ascii").strip()
                except OSError:
                    label = candidate.stem
                sensors.append((candidate, f"{name}:{label}"))
        return tuple(sensors)

    async def start(self) -> Result[EnergyMeasurementSession, ActionFailure]:
        rapl = self._rapl_readings()
        if rapl:
            return Success(_RaplSession(rapl, time.perf_counter()))
        power_sensors = self._power_sensors()
        if power_sensors:
            return Success(
                _PowerSensorSession(
                    tuple(path for path, _scope in power_sensors),
                    tuple(scope for _path, scope in power_sensors),
                    self.sample_interval_seconds,
                )
            )
        return Failure(
            make_assistant_failure(
                code="ENERGY_UNAVAILABLE",
                message="no readable Linux RAPL or hwmon power sensor is available",
            )
        )
