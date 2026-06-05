import sys
from pathlib import Path


def import_flashinfer():
    """Import FlashInfer, preferring a sibling source checkout when present.

    In this workspace the repository root may also contain a top-level
    `flashinfer/` source checkout. Importing `flashinfer` from the workspace
    root would otherwise resolve to an empty namespace package instead of the
    actual Python package living at `flashinfer/flashinfer/`.
    """
    try:
        import flashinfer  # type: ignore
        if hasattr(flashinfer, "BatchPrefillWithPagedKVCacheWrapper"):
            return flashinfer
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    local_checkout = repo_root / "flashinfer"
    init_py = local_checkout / "flashinfer" / "__init__.py"
    if init_py.exists():
        checkout_str = str(local_checkout)
        if checkout_str not in sys.path:
            sys.path.insert(0, checkout_str)
        stale = sys.modules.get("flashinfer")
        if stale is not None and getattr(stale, "__file__", None) is None:
            sys.modules.pop("flashinfer", None)
        import flashinfer  # type: ignore
        return flashinfer

    raise ImportError("flashinfer is not available")
