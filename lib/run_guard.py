# -*- coding: utf-8 -*-
"""Single-instance lock and GPU headroom check for training runs.

This module contains no forward or backward computation and does not affect any
numerical result. It provides two safeguards that run before training starts:

  1) SingleInstance       - only one training process may use a given log
                            directory (implemented with a file lock).
  2) check_gpu_headroom   - verify that the GPU has enough free memory at launch
                            time instead of failing later inside an arbitrary
                            kernel once the device is shared with another job.

Two processes training into the same directory also truncate each other's log,
because logging is opened with filemode='w'. The lock prevents that as well.

Usage (in main() of train.py, before logging.basicConfig(filemode='w')):

    from lib.run_guard import acquire_or_exit, check_gpu_headroom
    _guard = acquire_or_exit(opt.model_name, enabled=opt.single_instance,
                             min_free_mb=opt.min_free_gpu_mb,
                             gpu_id=opt.gpu_id)
"""

import atexit
import os
import sys
import time

_LOCK_FD = None
_LOCK_PATH = None
_LOCK_NAME = '.train.lock'

# Automatic threshold: refuse to start when free memory is below this fraction of
# the total device memory.
_AUTO_FREE_RATIO = 0.60


# ---------------------------------------------------------------- file lock
def _hostname():
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return ''


def _ensure_one_byte(fd):
    """Windows _locking needs a non-empty file to lock at the current offset."""
    if os.fstat(fd).st_size == 0:
        os.write(fd, b'\0')
    os.lseek(fd, 0, os.SEEK_SET)


def _try_lock(fd):
    if os.name == 'nt':
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _read_holder(path):
    """Read the holder description written by the process owning the lock.

    Byte 0 is the lock byte, and on Windows a locked region cannot even be read,
    so the description always starts at byte 1.
    """
    try:
        with open(path, 'rb') as f:
            f.seek(1)
            txt = f.read().decode('utf-8', 'replace').strip().strip('\x00').strip()
        return txt or '(no holder information recorded)'
    except Exception as exc:
        return '(failed to read lock file: %s)' % exc


def release():
    """Release the lock; called by atexit, no manual call is required."""
    global _LOCK_FD, _LOCK_PATH
    if _LOCK_FD is None:
        return
    try:
        if os.name != 'nt':
            import fcntl
            fcntl.flock(_LOCK_FD, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(_LOCK_FD)
    except Exception:
        pass
    _LOCK_FD = None


def _gpu_processes():
    """List other processes holding the GPU (diagnostic only; empty on failure)."""
    if os.name == 'nt':
        return []
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'],
            stderr=subprocess.DEVNULL, timeout=10).decode('utf-8', 'ignore')
        me = os.getpid()
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            if int(parts[0]) == me:
                continue
            rows.append('  PID %-8s %-12s %s' % (parts[0], parts[2],
                                                 os.path.basename(parts[1])))
        return rows
    except Exception:
        return []


def _free_mb(device):
    import torch
    free, total = torch.cuda.mem_get_info(int(device))
    return free // (1024 * 1024), total // (1024 * 1024)


def check_gpu_headroom(gpu_id=0, min_free_mb=0):
    """Check free GPU memory before training.

    min_free_mb:  0  = automatic (require free >= 60% of total memory)
                 >0  = hard threshold in MiB
                 <0  = disable the check, only report the current headroom
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free_mb, total_mb = _free_mb(gpu_id)
    except Exception as exc:
        print('[run_guard] cannot read GPU memory, check skipped: %s' % exc)
        return

    name = ''
    try:
        import torch
        name = torch.cuda.get_device_name(int(gpu_id))
    except Exception:
        pass

    need = (int(total_mb * _AUTO_FREE_RATIO) if int(min_free_mb) == 0
            else int(min_free_mb))

    print('[run_guard] GPU%d %s | free %d MiB / total %d MiB | need >= %d MiB'
          % (int(gpu_id), name, free_mb, total_mb, max(need, 0)))

    others = _gpu_processes()
    if others:
        print('[run_guard] other processes on this GPU:')
        for row in others:
            print(row)

    if int(min_free_mb) < 0 or free_mb >= need:
        return

    total_gib = total_mb / 1024.0
    sys.stdout.flush()          # keep log order when redirected by nohup
    print('', file=sys.stderr)
    print('=' * 72, file=sys.stderr)
    print('Refusing to start: not enough free GPU memory', file=sys.stderr)
    print('  total device memory : %.1f GiB' % total_gib, file=sys.stderr)
    print('  currently free      : %.1f GiB' % (free_mb / 1024.0), file=sys.stderr)
    print('  required by config  : >= %.1f GiB' % (max(need, 0) / 1024.0), file=sys.stderr)
    if others:
        print('  other processes     :', file=sys.stderr)
        for row in others:
            print('  ' + row.strip(), file=sys.stderr)
    print('', file=sys.stderr)
    print('This is almost always caused by a previous run that did not exit before a', file=sys.stderr)
    print('second one was launched. The second process then fails on a small kernel such', file=sys.stderr)
    print('as topk or masked_fill, which looks like an oversized model but is a conflict.', file=sys.stderr)
    print('', file=sys.stderr)
    print('How to resolve:', file=sys.stderr)
    print('  1) ps -eo pid,etime,cmd | grep train.py   and terminate the stale run', file=sys.stderr)
    print('  2) start again once the device is free', file=sys.stderr)
    print('  3) to deliberately share the device: --min_free_gpu_mb -1', file=sys.stderr)
    print('=' * 72, file=sys.stderr)
    sys.exit(4)


def acquire_or_exit(log_dir, enabled=1, name=_LOCK_NAME,
                    min_free_mb=0, gpu_id=0):
    """Acquire the single-instance lock, then verify GPU headroom.

    Exits when the lock is held by another process or the GPU is too full.
    Returns the file descriptor, or None when enabled=0.
    """
    global _LOCK_FD, _LOCK_PATH

    if not enabled:
        print('[run_guard] --single_instance 0: single-instance lock skipped')
        check_gpu_headroom(gpu_id, min_free_mb)
        return None

    try:
        if log_dir and not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except Exception:
        pass

    path = os.path.join(log_dir or '.', name)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    _ensure_one_byte(fd)

    if not _try_lock(fd):
        holder = _read_holder(path)
        os.close(fd)
        print('', file=sys.stderr)
        print('=' * 72, file=sys.stderr)
        print('[run_guard] another training process already uses %s'
              % os.path.abspath(log_dir), file=sys.stderr)
        print('  holder: %s' % holder, file=sys.stderr)
        print('', file=sys.stderr)
        print('Starting a second process in this directory would:', file=sys.stderr)
        print('  1) make both processes compete for the same GPU and OOM the newer one;', file=sys.stderr)
        print('  2) truncate the log of the first process via logging(filemode="w").', file=sys.stderr)
        print('', file=sys.stderr)
        print('Check whether the previous run is still needed:', file=sys.stderr)
        print('  ps -eo pid,etime,cmd | grep train.py', file=sys.stderr)
        print('Keep it -> do not start; rerun -> terminate it first.', file=sys.stderr)
        print('To run in parallel (different logger_name only): --single_instance 0', file=sys.stderr)
        print('=' * 72, file=sys.stderr)
        sys.exit(3)

    info = 'pid=%d  host=%s  start=%s  cwd=%s\n' % (
        os.getpid(), _hostname(),
        time.strftime('%Y-%m-%d %H:%M:%S'), os.getcwd())
    try:
        # Byte 0 is reserved for the lock; the description starts at byte 1 so that
        # other processes can read it without holding the lock.
        raw = info.encode('utf-8')
        os.lseek(fd, 1, os.SEEK_SET)
        os.write(fd, raw)
        os.ftruncate(fd, 1 + len(raw))
    except Exception:
        pass

    _LOCK_FD, _LOCK_PATH = fd, path
    atexit.register(release)
    print('[run_guard] training lock acquired: %s (pid=%d)' % (path, os.getpid()))

    check_gpu_headroom(gpu_id, min_free_mb)
    return fd
