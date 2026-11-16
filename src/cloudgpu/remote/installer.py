"""Base AppInstaller ABC. Stdlib-only."""

from __future__ import annotations

import abc
import functools
import os
import shutil

from . import instance
from .state import State
from .utils import log

# The methods that change the host; each is guarded in every subclass that defines it.
GUARDED = ("install", "recover")

# The status an already-recorded app gets when installing it again fails: the next recover
# installs it again rather than only recovering it (remote/recovery.py).
INSTALL_FAILED = "install_failed"


def _guarded(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        instance.require_instance()  # before the first host-changing command
        return method(self, *args, **kwargs)
    return wrapper


def _marks_failed_install(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.mark_install_failed()
            raise
    return wrapper


class AppInstaller(abc.ABC):
    """Abstract base class for app installers."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "install" in cls.__dict__:
            cls.install = _marks_failed_install(cls.__dict__["install"])
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

    def mark_install_failed(self) -> None:
        """An install failed: if the app was recorded, say so in its record (keeping the rest,
        a re-provision mark included), so the next recover installs it again. An app installed
        for the first time stays unrecorded."""
        record = self.state.get_app(self.name)
        if record is not None:
            self.state.set_app(self.name, {**record, "status": INSTALL_FAILED})


CLONE_SUFFIX = ".clone"


def restored_names(sync_paths, app: str) -> frozenset[str]:
    """The names directly inside ``apps/<app>`` that a restore from the bucket may create."""
    prefix = f"apps/{app}/"
    return frozenset(p[len(prefix):].split("/")[0] for p in sync_paths if p.startswith(prefix))


def needs_clone(app_dir: str, restored: frozenset[str]) -> bool:
    """Whether ``app_dir`` still has to be cloned.

    Not only when it is missing: a restore from the bucket creates ``<app dir>/output`` and
    the like before any install has run, so a directory holding nothing but those names is
    not an install. Nor is one with an unfinished clone beside it (see ``clone_repo``).
    """
    if not os.path.isdir(app_dir):
        return not os.path.lexists(app_dir)
    if os.path.lexists(app_dir.rstrip("/") + CLONE_SUFFIX):
        return True
    names = os.listdir(app_dir)
    return bool(names) and all(name in restored for name in names)


def has_checkout(app_dir: str) -> bool:
    return os.path.isdir(os.path.join(app_dir, ".git"))


def _move_in(src: str, dst: str, skip: tuple[str, ...] = ()) -> None:
    """Move what is in ``src`` into ``dst``; whatever ``dst`` already has stays as it is."""
    for name in sorted(os.listdir(src)):
        if name in skip:
            continue
        source, target = os.path.join(src, name), os.path.join(dst, name)
        if not os.path.lexists(target):
            os.rename(source, target)
        elif (os.path.isdir(source) and not os.path.islink(source)
              and os.path.isdir(target) and not os.path.islink(target)):
            try:
                _move_in(source, target)
            except OSError as e:
                # Only the repo's own files under a restored directory: not worth failing for.
                log(f"Left some of the repo's files out of {target}: {e.strerror or e}", "warn")


def clone_repo(repo: str, app_dir: str, run) -> None:
    """``git clone repo app_dir``, also when ``app_dir`` already holds restored directories.

    git won't clone into a directory with anything in it, so the clone goes beside it and
    is moved in: a file already there (the person's, restored) wins over the repo's, and
    the clone beside it is removed last, so an interrupted move is seen and done again.
    """
    if not os.path.lexists(app_dir):
        run(["git", "clone", repo, app_dir])
        return
    if os.path.islink(app_dir) or not os.path.isdir(app_dir):
        raise RuntimeError(f"{app_dir} is in the way of the install: it is not a directory.")
    staging = app_dir.rstrip("/") + CLONE_SUFFIX
    if os.path.islink(staging):
        os.remove(staging)
    shutil.rmtree(staging, ignore_errors=True)
    run(["git", "clone", repo, staging])
    if not has_checkout(staging):
        raise RuntimeError(f"git clone of {repo} left no checkout at {staging}.")
    _move_in(staging, app_dir, skip=(".git",))
    if os.path.lexists(os.path.join(app_dir, ".git")):  # from a move that was interrupted
        shutil.rmtree(os.path.join(app_dir, ".git"), ignore_errors=True)
    os.rename(os.path.join(staging, ".git"), os.path.join(app_dir, ".git"))
    shutil.rmtree(staging, ignore_errors=True)
