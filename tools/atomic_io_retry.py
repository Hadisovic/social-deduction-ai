"""Bounded retry for transient Windows sharing violations; no data transformations."""
import time


def retry_io(operation,*args,attempts=8,delay=.05,**kwargs):
    for attempt in range(attempts):
        try:return operation(*args,**kwargs)
        except PermissionError:
            if attempt==attempts-1:raise
            time.sleep(min(delay*(attempt+1),.25))
