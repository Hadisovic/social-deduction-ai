"""Short-lived JSON reads; producers separately retry Windows reader locks."""
from contextlib import contextmanager
import json
from pathlib import Path
from time import sleep


@contextmanager
def open_shared(path):
    with Path(path).open('r',encoding='utf-8') as stream:yield stream


def read_json(path):
    with open_shared(path) as stream:return json.load(stream)


def write_json(path,record,attempts=8):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(record,allow_nan=False))
    for attempt in range(attempts):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt+1==attempts:raise
            sleep(.003)
