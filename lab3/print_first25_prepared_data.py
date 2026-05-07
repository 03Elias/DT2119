import os
import numpy as np


def format_row(values, max_items=8):
    flat = np.asarray(values).ravel()
    shown = flat[:max_items]
    text = np.array2string(shown, precision=4, separator=", ")
    if flat.size > max_items:
        text = text[:-1] + ", ...]"
    return text


def print_array_preview(name, array, n=25):
    print("=" * 80)
    print(f"{name}: shape={array.shape}, dtype={array.dtype}")

    rows = min(n, len(array)) if array.ndim > 0 else 1
    print(f"Showing first {rows} element(s)")

    if array.ndim == 1:
        for i in range(rows):
            print(f"[{i}] {array[i]}")
        return

    for i in range(rows):
        print(f"[{i}] {format_row(array[i])}")


def print_split_preview(data, split_name, feature_name, n=25):
    x_key = f"{feature_name}_{split_name}_x"
    y_key = f"{split_name}_y"

    if x_key not in data or y_key not in data:
        print("=" * 80)
        print(f"Missing keys for split '{split_name}' and feature '{feature_name}'.")
        return

    x = data[x_key]
    y = data[y_key]

    print("=" * 80)
    print(f"Split: {split_name}, feature: {feature_name}")
    print(f"Features: shape={x.shape}, dtype={x.dtype}")
    print(f"Targets: shape={y.shape}, dtype={y.dtype}")
    print(f"Showing first {min(n, len(x))} frame(s)")

    for i in range(min(n, len(x))):
        print(f"[{i}] x={format_row(x[i])} | y={y[i]}")


def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    npz_path = os.path.join(base_dir, "prepared_data.npz")

    if not os.path.exists(npz_path):
        print(f"Missing file: {npz_path}")
        return

    with np.load(npz_path, allow_pickle=True) as data:
        print(f"File: {npz_path}")
        print("Keys:", ", ".join(sorted(data.files)))

        print_array_preview("stateList", data["stateList"], n=25)

        for feature_name in ("lmfcc", "mspec", "dlmfcc", "dmspec"):
            print_split_preview(data, "train", feature_name, n=25)


if __name__ == "__main__":
    main()