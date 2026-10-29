"""Process lock shared by the trainer and post-interruption reporter; no Torch."""
from contextlib import contextmanager
from pathlib import Path
import os


@contextmanager
def run_lock(folder):
    path=Path(folder)/'writer.lock';handle=path.open('a+b');handle.seek(0)
    if path.stat().st_size==0:handle.write(b'0');handle.flush();handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('Another training process owns this run folder')
    try:yield
    finally:handle.close()
