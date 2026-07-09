"""No-op wandb shim for local evaluation runs.

Some optional dependencies import wandb at module import time even when logging
is disabled. The local nuPlan environment has a wandb/protobuf mismatch, so
quick-test scripts can prepend this module on PYTHONPATH to make those optional
imports harmless. Set PLUTO_EVAL_ALLOW_WANDB=1 to use the real wandb package.
"""

run = None
config = {}


def init(*args, **kwargs):
    return None


def log(*args, **kwargs):
    return None


def finish(*args, **kwargs):
    return None


def require(*args, **kwargs):
    return None


class _NoOp:
    def __init__(self, *args, **kwargs):
        pass


Image = _NoOp
Audio = _NoOp
Table = _NoOp
Artifact = _NoOp
