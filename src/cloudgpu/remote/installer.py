"""Base AppInstaller ABC. Stdlib-only."""

from __future__ import annotations

import abc
from .state import State


class AppInstaller(abc.ABC):
    """Abstract base class for app installers."""

    def __init__(self, state: State):
        self.state = state

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """App name (e.g., 'comfyui')."""
        ...

    @abc.abstractmethod
    def install(self) -> None:
        """Install the app from scratch."""
        ...

    @abc.abstractmethod
    def verify(self) -> bool:
        """Verify the app is properly installed and functional."""
        ...

    @abc.abstractmethod
    def recover(self) -> None:
        """Recover/repair the app on a new instance."""
        ...

    @abc.abstractmethod
    def get_status(self) -> dict:
        """Get current status of the app."""
        ...
