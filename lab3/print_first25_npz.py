import os
import numpy as np


def load_items(npz_path):
    """Load utterance items from a known npz key."""
    with np.load(npz_path, allow_pickle=True) as npz:
        for key in ("data", "traindata", "testdata"):
            if key in npz:
                return npz[key], key
    raise KeyError("No supported data key found (expected one of: data, traindata, testdata)")


def summarize_item(item, idx):
    """Create a compact printable summary for one utterance item dict."""
    if isinstance(item, dict):
        filename = item.get("filename", "<no filename>")
        lmfcc_shape = getattr(item.get("lmfcc", None), "shape", None)
        mspec_shape = getattr(item.get("mspec", None), "shape", None)
        targets_len = len(item.get("targets", [])) if "targets" in item else None
        return (
            f"[{idx}] filename={filename}, "
            f"lmfcc_shape={lmfcc_shape}, "
            f"mspec_shape={mspec_shape}, "
            f"targets_len={targets_len}"
        )
    return f"[{idx}] {item}"


def print_first_n(npz_path, n=25):
    print("=" * 80)
    print(f"File: {npz_path}")

    if not os.path.exists(npz_path):
        print("Missing file.")
        return

    try:
        items, key = load_items(npz_path)
    except Exception as exc:
        print(f"Could not load items: {exc}")
        return

    total = len(items)
    print(f"Data key: {key}")
    print(f"Total items: {total}")
    print(f"Showing first {min(n, total)} items")

    for i, item in enumerate(items[:n]):
        print(summarize_item(item, i))


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, "traindata.npz")
    test_path = os.path.join(base_dir, "testdata.npz")

    print_first_n(train_path, n=25)
    print_first_n(test_path, n=25)
