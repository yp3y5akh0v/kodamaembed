"""Model download progress for terminals and logs."""

from __future__ import annotations

from typing import TextIO


def format_size(count: int) -> str:
    """Use decimal units to match the sizes on model download pages."""
    if count < 1000:
        return f"{count} B"
    value = count / 1000
    for unit in ("kB", "MB"):
        if value < 999.95:
            return f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"


class ProgressDisplay:
    """Redraw progress on terminals and limit log output to file starts and completions."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._terminal = stream.isatty()
        self._announced: tuple[str, str] | None = None
        self._drawn = ""

    def __call__(self, repository: str, file: str, received: int, total: int) -> None:
        name = f"{repository}/{file} ({format_size(total)})"
        if received >= total:
            self._print(f"Downloaded {name}")
            self._announced = None
        elif self._terminal:
            self._draw(f"Downloading {name} {received * 100 // total}%")
        elif self._announced != (repository, file):
            self._print(f"Downloading {name}")
            self._announced = (repository, file)

    def _draw(self, line: str) -> None:
        if line != self._drawn:
            self._stream.write("\r" + line.ljust(len(self._drawn)))
            self._stream.flush()
            self._drawn = line

    def _print(self, line: str) -> None:
        start = "\r" if self._drawn else ""
        self._stream.write(start + line.ljust(len(self._drawn)) + "\n")
        self._stream.flush()
        self._drawn = ""
