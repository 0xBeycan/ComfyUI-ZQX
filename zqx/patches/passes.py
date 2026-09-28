"""Pass context shared by all ZQX model patches.

Several patches run extra forward passes inside one model call (reference capture on the reference image,
steering towards/away prompts, PAG's perturbed pass, LoRA guidance's base pass).  When patches are chained,
the hooks of one patch also fire during another patch's extra passes.  To keep every patch's state tied to
the passes it is meant for, each ZQX DIFFUSION_MODEL wrapper pushes entries on a thread-local stack:

  ("enter",   owner, args, kwargs)   around the whole wrapper call
  ("foreign", owner, args, kwargs)   an extra pass on *different inputs* (other image / other prompt)
  ("variant", owner, args, kwargs)   an extra pass on the *same inputs* with a modified model

Rules used by the hooks (see `blocked`):
  * capturing features (reference K/V, steering activations) only in passes that are neither foreign nor a
    variant of another patch  -> each wrapper contributes exactly one "plain" pass
  * injecting (reference attention, steering, PAG identity attention) in plain and variant passes, never in
    another patch's foreign pass  -> a perturbed/base pass sees the same model as the plain pass
Hooks that need the token layout of the *current* forward (spatial LoRA) read the innermost entry's args.
"""
from __future__ import annotations

import contextlib
import threading

_local = threading.local()


def _stack():
    s = getattr(_local, "stack", None)
    if s is None:
        s = []
        _local.stack = s
    return s


@contextlib.contextmanager
def entry(kind: str, owner, args=(), kwargs=None):
    if kind not in ("enter", "foreign", "variant"):
        raise ValueError(kind)
    s = _stack()
    s.append((kind, owner, args, kwargs or {}))
    try:
        yield
    finally:
        s.pop()


def blocked(owner, kinds=("foreign", "variant")) -> bool:
    """True if, after `owner`'s innermost 'enter', another patch pushed an entry of one of `kinds`."""
    s = _stack()
    start = None
    for i in range(len(s) - 1, -1, -1):
        if s[i][0] == "enter" and s[i][1] is owner:
            start = i
            break
    if start is None:
        return True       # not inside this patch's wrapper at all
    return any(k in kinds and o is not owner for (k, o, _, _) in s[start + 1:])


def in_foreign_pass() -> bool:
    """True while any patch is running a foreign pass (different image / prompt).  Activation-level patches
    (reference attention, steering, PAG) stay passive there whatever the install order; model-defining patches
    (runtime / spatial LoRA, LoRA guidance, CADS) keep acting so that the extra pass sees the same model."""
    return any(k == "foreign" for (k, _, _, _) in _stack())


def current_call():
    """(args, kwargs, depth) of the innermost pushed entry, i.e. the arguments of the forward now running."""
    s = _stack()
    if not s:
        return None, None, 0
    _, _, a, k = s[-1]
    return a, k, len(s)


def call(executor, args, kwargs):
    """Run the next wrapper / the model after clearing a stale transformer_options['block_index'].

    Qwen-Image and Z-Image set block_index only inside their main block loop; Z-Image's refiner blocks run
    before it.  When a wrapper launches several forwards, the second forward's refiner calls would otherwise
    still see the last block index of the previous forward and be mistaken for main-block calls."""
    for obj in list(args) + list(kwargs.values()):
        if isinstance(obj, dict) and "block_index" in obj:
            obj.pop("block_index", None)
    return executor(*args, **kwargs)
