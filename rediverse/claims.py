"""Atomic per-item claims so several workers can share one output directory.

O_CREAT|O_EXCL is atomic on the filesystem: exactly one worker wins a claim, without a
coordinator. Two flavours:

  claim(out_path)               the item's own output file is the completion marker; a claim
                                whose output has not appeared within `expiry` seconds belonged
                                to a worker that stopped and is retaken
  claim_tag / release_tag       a claim held by "host pid" for the duration of a unit of work
                                (e.g. all samples of one prompt); a claim whose owner process
                                is gone is taken over

Standard library only, so it imports in every environment.
"""
from __future__ import annotations

import errno
import os
import socket
import time

EXPIRY_S = 3 * 3600


def claim(out_path: str, expiry: float = EXPIRY_S) -> bool:
    """Claim the item whose result is `out_path` (lock in <dir>/.claims/<name>.lock)."""
    d = os.path.join(os.path.dirname(out_path), ".claims")
    os.makedirs(d, exist_ok=True)
    lock = os.path.join(d, os.path.basename(out_path) + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) > expiry and not os.path.exists(out_path):
                os.utime(lock, None); return True
        except FileNotFoundError:
            pass
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno != errno.ESRCH      # EPERM = alive but not ours
    return True


def claim_tag(claim_dir: str, tag: str) -> bool:
    """Claim unit `tag` (file <claim_dir>/<tag>.claim holding "host pid"). True = ours to do.
    A claim left by a process that no longer exists on this host is taken over."""
    os.makedirs(claim_dir, exist_ok=True)
    f = os.path.join(claim_dir, f"{tag}.claim")
    me = f"{socket.gethostname()} {os.getpid()}"
    for _ in range(2):
        try:
            fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as fh:
                fh.write(me)
            return True
        except FileExistsError:
            try:
                host, pid = open(f).read().split()
            except (ValueError, OSError):
                return False               # being written right now; leave it alone
            if host != socket.gethostname() or _pid_alive(int(pid)):
                return False               # a live worker owns it
            try:
                os.unlink(f)               # its owner is gone: take it over
            except OSError:
                return False
    return False


def release_tag(claim_dir: str, tag: str) -> None:
    """Drop the claim on `tag` once all of its outputs are on disk."""
    try:
        os.unlink(os.path.join(claim_dir, f"{tag}.claim"))
    except OSError:
        pass
