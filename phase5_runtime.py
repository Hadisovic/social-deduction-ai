"""Deploy checkpoints with the mechanical execution protocol used in training."""
import hashlib
from pathlib import Path
from phase5_options import environment_for


def checked_environment(metadata, **kwargs):
    if 'options_sha256' in metadata:
        source = Path(__file__).with_name('phase5_options.py').read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(source).hexdigest() != metadata['options_sha256']:
            raise ValueError('Checkpoint runtime mismatch: phase5_options.py')
    return environment_for(metadata, **kwargs)
