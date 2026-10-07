"""Ultralytics 8.3.138 integration for AGM-YOLO.

The integration patches the parser's module aliases at runtime instead of
modifying site-packages. Custom classes are drop-in subclasses of the original
YOLO11 modules, so baseline state-dict keys remain compatible with yolo11n.pt.
"""
from __future__ import annotations

import os
from typing import Sequence

from torch import nn

from .core import AIC, GCE, MSA

PINNED_ULTRALYTICS = "8.3.138"

try:
    import ultralytics
    from ultralytics.nn.modules import C2PSA as _C2PSA
    from ultralytics.nn.modules import C3k2 as _C3k2
    from ultralytics.nn.modules import Detect as _Detect
except Exception as exc:  # pragma: no cover - gives a clean install error
    raise ImportError(
        "AGM-YOLO requires ultralytics==8.3.138. Install dependencies first."
    ) from exc


class C3k2AGM(_C3k2):
    """YOLO11 C3k2 with optional AIC applied to its output.

    Signature intentionally extends the upstream C3k2 signature so the original
    Ultralytics parser can inject c1/c2/repeats without a custom parse_model.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        aic: bool = False,
        reduction: int = 16,
    ):
        super().__init__(c1, c2, n=n, c3k=c3k, e=e, g=g, shortcut=shortcut)
        self.aic = AIC(c2, reduction=reduction) if aic else nn.Identity()

    def forward(self, x):
        return self.aic(super().forward(x))


class C2PSAAGM(_C2PSA):
    """YOLO11 C2PSA with optional GCE after the original block."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        gce: bool = False,
        reduction: int = 16,
    ):
        super().__init__(c1, c2, n=n, e=e)
        self.gce = GCE(c2, reduction=reduction) if gce else nn.Identity()

    def forward(self, x):
        return self.gce(super().forward(x))


class DetectAGM(_Detect):
    """YOLO11 Detect head with optional pre-detection MSA on P3/P4/P5.

    Ultralytics appends the input-channel list to YAML args. This constructor
    accepts both the stock form Detect(nc, ch) and AGM form Detect(nc, msa, ch),
    which keeps parser patching safe for baseline models as well.
    """

    def __init__(self, nc: int = 80, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            use_msa, ch = False, args[0]
        elif len(args) == 2:
            use_msa, ch = bool(args[0]), args[1]
        else:
            raise TypeError(
                "DetectAGM expects Detect(nc, ch) or Detect(nc, use_msa, ch); "
                f"received nc={nc}, args={args!r}"
            )
        ch = tuple(int(c) for c in ch)
        if len(ch) != 3:
            raise ValueError(f"AGM-YOLO expects three detection scales, got channels={ch}")
        super().__init__(nc=nc, ch=ch)
        self.use_msa = use_msa
        if use_msa:
            p3, p4, p5 = ch
            self.msa_p3 = MSA([p3, p4])
            self.msa_p4 = MSA([p4, p3, p5])
            self.msa_p5 = MSA([p5, p4])

    def forward(self, x: Sequence):
        if self.use_msa:
            if len(x) != 3:
                raise ValueError(f"MSA detection head expects P3/P4/P5, got {len(x)} inputs")
            p3, p4, p5 = x
            x = [
                self.msa_p3([p3, p4]),
                self.msa_p4([p4, p3, p5]),
                self.msa_p5([p5, p4]),
            ]
        return super().forward(list(x))


_PATCHED = False
_ORIGINALS = {}


def patch_ultralytics(strict_version: bool = True) -> None:
    """Patch Ultralytics parser aliases with AGM-compatible subclasses.

    Call this before constructing a YOLO model from an AGM YAML file. The patch
    is process-local and does not edit the installed Ultralytics package.
    """
    global _PATCHED
    if strict_version and ultralytics.__version__ != PINNED_ULTRALYTICS:
        allow = os.getenv("AGM_ALLOW_VERSION_MISMATCH", "0") == "1"
        if not allow:
            raise RuntimeError(
                f"Expected ultralytics=={PINNED_ULTRALYTICS}, found {ultralytics.__version__}. "
                "Install the pinned version, or set AGM_ALLOW_VERSION_MISMATCH=1 to override the version check."
            )
    if _PATCHED:
        return

    import ultralytics.nn.tasks as tasks

    for name, replacement in {
        "C3k2": C3k2AGM,
        "C2PSA": C2PSAAGM,
        "Detect": DetectAGM,
    }.items():
        _ORIGINALS[name] = getattr(tasks, name)
        setattr(tasks, name, replacement)
    _PATCHED = True


def unpatch_ultralytics() -> None:
    """Restore original parser aliases. Normally unnecessary for training scripts."""
    global _PATCHED
    if not _PATCHED:
        return
    import ultralytics.nn.tasks as tasks

    for name, original in _ORIGINALS.items():
        setattr(tasks, name, original)
    _PATCHED = False
