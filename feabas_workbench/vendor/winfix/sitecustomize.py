"""
Run-time fixes for FEABAS 3.0.5, applied without touching the FEABAS installation.

The workbench puts this folder on PYTHONPATH for every FEABAS process (including
multiprocessing children, which inherit the environment), so the patches apply wherever
the affected module is imported.

1. ``stitching_matcher`` never binds ``phtm`` (Windows and Linux alike). FEABAS defaults
   ``compute_photometric`` to False, which skips the only assignments to ``phtm``, but
   then returns it unconditionally - so every tile pair that passes the coarse confidence
   check dies with "cannot access local variable 'phtm'" and stitch matching produces no
   matches at all. Patched by defaulting the flag to True so the variable is always bound.

2. Windows only: FEABAS builds kvstore URLs as ``'file://' + 'D:/path/'`` which TensorStore
   rejects ("Invalid file path"); the accepted form is ``file:///D:/path/``. Rewrites file
   URLs in specs passed to tensorstore.open / tensorstore.Spec / tensorstore.KvStore.open
   and makes ``KvStore.url`` report the FEABAS form again, so FEABAS's own ``'file://'``
   stripping keeps producing valid Windows paths.
"""

import os
import re
import sys

_DRIVE_URL = re.compile(r"^file://([A-Za-z]:/)")


def _fix(obj):
    if isinstance(obj, str):
        return _DRIVE_URL.sub(r"file:///\1", obj)
    if isinstance(obj, dict):
        return {k: _fix(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix(v) for v in obj]
    return obj


def _unfix(url: str) -> str:
    return re.sub(r"^file:///([A-Za-z]:/)", r"file://\1", url)


def _install():
    if os.name != "nt":
        return
    try:
        import tensorstore as ts
    except Exception:  # noqa: BLE001
        return
    if getattr(ts, "_feabas_workbench_winfix", False):
        return
    _open = ts.open
    _spec = ts.Spec
    _kv_open = ts.KvStore.open
    _kv_spec = ts.KvStore.Spec

    def open_(spec, *a, **kw):
        return _open(_fix(spec), *a, **kw)

    def spec_(json, *a, **kw):
        return _spec(_fix(json), *a, **kw)

    def kv_open(spec, *a, **kw):
        return _kv_open(_fix(spec), *a, **kw)

    def kv_spec(json, *a, **kw):
        return _kv_spec(_fix(json), *a, **kw)

    ts.open = open_
    ts.Spec = spec_
    ts.KvStore.open = staticmethod(kv_open)
    ts.KvStore.Spec = staticmethod(kv_spec)
    # KvStore.url is a C-level property on a non-heap type; wrap the class attribute if possible
    try:
        orig_url = ts.KvStore.url

        class _Url:
            def __get__(self, inst, owner):
                if inst is None:
                    return orig_url
                return _unfix(orig_url.__get__(inst, owner))

        ts.KvStore.url = _Url()
    except (AttributeError, TypeError):
        pass
    ts._feabas_workbench_winfix = True


def _install_matcher():
    """Make feabas.matcher.stitching_matcher always bind ``phtm`` (see 1. above)."""
    try:
        from feabas import matcher
    except Exception:  # noqa: BLE001
        return
    if getattr(matcher, "_feabas_workbench_phtm_fix", False):
        return
    orig = getattr(matcher, "stitching_matcher", None)
    if orig is None:
        return

    def stitching_matcher(img0, img1, **kwargs):
        # True takes the branch that assigns phtm; the photometric tuple it returns is
        # what FEABAS's own caller already expects (stitcher.py handles None and tuple).
        kwargs.setdefault("compute_photometric", True)
        return orig(img0, img1, **kwargs)

    stitching_matcher.__doc__ = orig.__doc__
    stitching_matcher.__name__ = orig.__name__
    matcher.stitching_matcher = stitching_matcher
    matcher._feabas_workbench_phtm_fix = True
    # feabas.stitcher imports the name directly, so rebind it there too if already loaded.
    stitcher = sys.modules.get("feabas.stitcher")
    if stitcher is not None and hasattr(stitcher, "stitching_matcher"):
        stitcher.stitching_matcher = stitching_matcher


class _Finder:
    """Apply a patch right after the module it targets is imported for the first time."""

    def __init__(self, target, apply):
        self.target, self.apply = target, apply

    def find_spec(self, name, path=None, target=None):
        if name != self.target:
            return None
        import importlib.util
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.loader is not None:
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                self.apply()

            spec.loader.exec_module = exec_module
        return spec


def _hook(module, apply):
    if module in sys.modules:
        apply()
    else:
        sys.meta_path.insert(0, _Finder(module, apply))


_hook("feabas.matcher", _install_matcher)
if os.name == "nt":
    _hook("tensorstore", _install)
