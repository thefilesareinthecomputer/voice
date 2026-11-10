"""The state folder and the pid locks that let a later command find the one running.

A pid file is trusted only while its writer holds an flock on it. The kernel drops the
lock when that process exits, however it exits, so the pid of a process that has exited
is never signalled. What remains is the instant between reading a live holder and
signalling it: the holder would have to exit and its pid be reused inside that instant.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Mapping
from pathlib import Path


def xdg(env: Mapping[str, str], var: str, default: str) -> Path:
    """An XDG base folder. The variable counts only when absolute, as the XDG spec says; a relative
    one would put files in whatever folder a command ran from."""
    base = env.get(var, "")
    return Path(base) if os.path.isabs(base) else Path.home() / default


STATE = xdg(os.environ, "XDG_STATE_HOME", ".local/state") / "claude-voice"


def folder() -> Path:
    """The state folder, private to this user. A link in its place is refused, so the folder's
    mode change and every file in it stay where they were meant to be."""
    if STATE.is_symlink():
        raise OSError(f"{STATE} is a link; voice keeps its state only in a real folder")
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    STATE.chmod(0o700)  # also closes a folder an older version left readable
    return STATE


def _open(path: Path) -> int:
    folder()
    return os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)


def hold(path: Path, wait: bool = False) -> int | None:
    """Take the lock on `path` and empty it; the caller then writes the pid with `write_pid`.
    Returns the descriptor, which keeps the lock until every copy of it is closed (a child
    started with it in `pass_fds` shares it), or None when another process holds it."""
    fd = _open(path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    return fd


def write_pid(fd: int, pid: int) -> None:
    os.pwrite(fd, str(pid).encode(), 0)


def locked(path: Path) -> bool:
    """True while any live process holds the lock on `path`, whether or not it has written its pid."""
    try:
        fd = _open(path)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def holder(path: Path) -> int | None:
    """The pid in `path` while a live process holds its lock; None when nobody does, while the
    holder has not written its pid yet, or when the file names no ordinary process (1 or less,
    which os.kill would treat as launchd, a process group or every process)."""
    try:
        fd = _open(path)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            pid = int(os.pread(fd, 32, 0).decode().strip())
        except ValueError:
            return None
        return pid if pid > 1 else None
    finally:
        os.close(fd)
    return None  # the lock was free, so whatever pid the file names is gone
