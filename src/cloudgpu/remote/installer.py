"""Base AppInstaller ABC. Stdlib-only."""

from __future__ import annotations

import abc
import functools

from . import instance
from .state import State

# The methods that change the host; each is guarded in every subclass that defines it.
GUARDED = ("install", "recover")


def _guarded(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        instance.require_instance()  # before the first host-changing command
        return method(self, *args, **kwargs)
    return wrapper


class AppInstaller(abc.ABC):
    """Abstract base class for app installers."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in GUARDED:
            if name in cls.__dict__:
                setattr(cls, name, _guarded(cls.__dict__[name]))

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

    def service_spec(self) -> dict | None:
        """Return a systemd service spec for a long-running server app, or None.

        Apps that should run as a managed service override this to return
        {"name", "exec_start", "workdir", "port"}. Default: not a service.
        """
        return None

    def save_record(self, record: dict) -> None:
        """Replace the app's record in state.json, keeping a re-provision mark it carries."""
        if (self.state.get_app(self.name) or {}).get("reprovision"):
            record = {**record, "reprovision": True}
        self.state.set_app(self.name, record)
