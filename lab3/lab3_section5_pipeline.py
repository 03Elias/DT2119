"""Local PyTorch pipeline for DT2119 Lab 3 Sections 5 and 5.1.

This script trains a small frame-wise DNN on the flattened arrays in
prepared_data.npz and evaluates the model on the validation and test sets.
It also computes the evaluation items requested in the lab handout:

* frame accuracy at the state level
* frame accuracy at the phoneme level
* confusion matrices at both levels
* edit-distance based PER at both levels after collapsing repeats
* posterior plot for one example utterance from testdata.npz

The implementation is intentionally simple and local-friendly:
CPU by default, CUDA when available, CrossEntropyLoss + Adam, and a
configurable feed-forward network with ReLU activations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dynamic_features import stack_context


FEATURE_KEYS = {
    "lmfcc": "lmfcc",
    "mspec": "mspec",
    "dlmfcc": "dlmfcc",
    "dmspec": "dmspec",
}


@dataclass
class ExperimentResult:
    feature_type: str
    hidden_layers: list[int]
    hidden_size: int
    epochs: int
    train_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    train_loss: float
    validation_loss: float
    test_loss: float
    state_accuracy: float
    phoneme_accuracy: float
    state_per: float
    phoneme_per: float
    model_path: str
    posterior_plot_path: str


class FeedForwardNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Sequence[int], output_size: int):
        super().__init__()

        layers: list[nn.Module] = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_hidden_sizes(values: Sequence[int] | None, hidden_size: int) -> list[int]:
    if values:
        return [int(v) for v in values]
    return [hidden_size]


def load_prepared_data(data_path: Path):
    with np.load(data_path, allow_pickle=True) as data:
        state_list = data["stateList"].tolist()
        arrays = {name: data[name] for name in data.files}
    return arrays, state_list


def standardize_with_saved_stats(features: np.ndarray, arrays: dict, feature_type: str) -> np.ndarray:
    mean = arrays.get(f"{feature_type}_scaler_mean")
    scale = arrays.get(f"{feature_type}_scaler_scale")
    if mean is None or scale is None:
        raise KeyError(f"Missing scaler statistics for feature type '{feature_type}'")

    standardized = (np.asarray(features, dtype=np.float32) - mean) / scale
    return standardized.astype(np.float32)


def get_feature_arrays(arrays: dict, feature_type: str):
    train_x = arrays[f"{feature_type}_train_x"]
    val_x = arrays[f"{feature_type}_val_x"]
    test_x = arrays[f"{feature_type}_test_x"]
    train_y = arrays["train_y"]
    val_y = arrays["val_y"]
    test_y = arrays["test_y"]
    return train_x, val_x, test_x, train_y, val_y, test_y


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    features = torch.from_numpy(np.asarray(x, dtype=np.float32))
    labels = torch.from_numpy(np.asarray(y, dtype=np.int64))
    dataset = TensorDataset(features, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = batch_y.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_examples += batch_size

    average_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
            }
        )

    return history


def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device):
    loader = build_loader(x, np.zeros(len(x), dtype=np.int64), batch_size=batch_size, shuffle=False, device=device)
    model.eval()

    logits_list = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits_list.append(model(batch_x).cpu())

    logits = torch.cat(logits_list, dim=0)
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1).numpy()
    return probabilities.numpy(), predictions


def state_to_phoneme_name(state_name: str) -> str:
    return state_name.rsplit("_", 1)[0]


def indices_to_phonemes(indices: Sequence[int], state_list: Sequence[str]) -> list[str]:
    return [state_to_phoneme_name(state_list[index]) for index in indices]


def collapse_consecutive(values: Sequence[str | int]) -> list[str | int]:
    collapsed = []
    previous = object()
    for value in values:
        if value != previous:
            collapsed.append(value)
            previous = value
    return collapsed


def edit_distance(reference: Sequence[str | int], hypothesis: Sequence[str | int]) -> int:
    n = len(reference)
    m = len(hypothesis)

    if n == 0:
        return m
    if m == 0:
        return n

    previous_row = list(range(m + 1))
    for i, ref_item in enumerate(reference, start=1):
        current_row = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_item == hyp_item else 1
            current_row.append(
                min(
                    previous_row[j] + 1,
                    current_row[j - 1] + 1,
                    previous_row[j - 1] + substitution_cost,
                )
            )
        previous_row = current_row

    return previous_row[-1]


def per_score(reference: Sequence[str | int], hypothesis: Sequence[str | int]) -> float:
    reference = collapse_consecutive(reference)
    hypothesis = collapse_consecutive(hypothesis)
    if not reference:
        return 0.0
    return edit_distance(reference, hypothesis) / len(reference)


def confusion_matrix(reference: Sequence[int], hypothesis: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for ref, hyp in zip(reference, hypothesis):
        matrix[int(ref), int(hyp)] += 1
    return matrix


def accuracy_score(reference: Sequence[int], hypothesis: Sequence[int]) -> float:
    reference = np.asarray(reference)
    hypothesis = np.asarray(hypothesis)
    return float((reference == hypothesis).mean()) if len(reference) else 0.0


def evaluate_predictions(predictions: np.ndarray, targets: np.ndarray, state_list: Sequence[str]):
    state_acc = accuracy_score(targets, predictions)
    state_cm = confusion_matrix(targets, predictions, len(state_list))

    ref_state_names = [state_list[index] for index in targets]
    hyp_state_names = [state_list[index] for index in predictions]
    ref_phonemes = indices_to_phonemes(targets, state_list)
    hyp_phonemes = indices_to_phonemes(predictions, state_list)

    ref_phoneme_indices = [state_to_phoneme_name(name) for name in ref_state_names]
    hyp_phoneme_indices = [state_to_phoneme_name(name) for name in hyp_state_names]

    phoneme_labels = sorted({state_to_phoneme_name(name) for name in state_list})
    phoneme_to_index = {label: index for index, label in enumerate(phoneme_labels)}
    ref_phoneme_ids = [phoneme_to_index[label] for label in ref_phoneme_indices]
    hyp_phoneme_ids = [phoneme_to_index[label] for label in hyp_phoneme_indices]
    phoneme_acc = accuracy_score(ref_phoneme_ids, hyp_phoneme_ids)
    phoneme_cm = confusion_matrix(ref_phoneme_ids, hyp_phoneme_ids, len(phoneme_labels))

    return {
        "state_accuracy": state_acc,
        "state_confusion_matrix": state_cm,
        "phoneme_accuracy": phoneme_acc,
        "phoneme_confusion_matrix": phoneme_cm,
        "phoneme_labels": phoneme_labels,
    }


def evaluate_per_on_utterances(
    model: nn.Module,
    utterances: Iterable[dict],
    arrays: dict,
    feature_type: str,
    state_list: Sequence[str],
    device: torch.device,
):
    total_state_distance = 0
    total_state_length = 0
    total_phoneme_distance = 0
    total_phoneme_length = 0

    for utterance_item in utterances:
        features, targets = prepare_example_features(utterance_item, arrays, feature_type)
        _, predictions = predict(model, features, batch_size=512, device=device)

        ref_state_names = [state_list[index] for index in targets]
        hyp_state_names = [state_list[index] for index in predictions]
        ref_phonemes = indices_to_phonemes(targets, state_list)
        hyp_phonemes = indices_to_phonemes(predictions, state_list)

        collapsed_ref_states = collapse_consecutive(ref_state_names)
        collapsed_hyp_states = collapse_consecutive(hyp_state_names)
        collapsed_ref_phonemes = collapse_consecutive(ref_phonemes)
        collapsed_hyp_phonemes = collapse_consecutive(hyp_phonemes)

        total_state_distance += edit_distance(collapsed_ref_states, collapsed_hyp_states)
        total_state_length += len(collapsed_ref_states)
        total_phoneme_distance += edit_distance(collapsed_ref_phonemes, collapsed_hyp_phonemes)
        total_phoneme_length += len(collapsed_ref_phonemes)

    state_per = total_state_distance / max(total_state_length, 1)
    phoneme_per = total_phoneme_distance / max(total_phoneme_length, 1)
    return state_per, phoneme_per


def prepare_example_features(utterance_item: dict, arrays: dict, feature_type: str):
    # For dynamic features, we generate them from the base feature on the fly
    # because testdata.npz only stores the raw lmfcc and mspec
    if feature_type.startswith("d"):
        base_feature = feature_type[1:]
        features = utterance_item[base_feature]
        features = stack_context(features)
    else:
        features = utterance_item[feature_type]
    
    features = standardize_with_saved_stats(features, arrays, feature_type)
    targets = np.asarray(utterance_item["targets"], dtype=np.int64)
    return features, targets


def plot_posteriors(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    state_list: Sequence[str],
    device: torch.device,
    output_path: Path,
):
    probabilities, predictions = predict(model, features, batch_size=512, device=device)

    frame_count = min(len(features), 300)
    plt.figure(figsize=(14, 6))
    plt.imshow(
        probabilities[:frame_count].T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    plt.colorbar(label="Posterior")
    plt.plot(np.asarray(targets[:frame_count]), color="white", linewidth=1.0, label="target state index")
    plt.plot(predictions[:frame_count], color="red", linewidth=1.0, alpha=0.8, label="predicted state index")
    plt.xlabel("Frame")
    plt.ylabel("State index / posterior matrix row")
    plt.title("Posterior probabilities for one example utterance")
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_example_utterance(testdata_path: Path, index: int = 0):
    with np.load(testdata_path, allow_pickle=True) as data:
        items = data["data"]
        return items[index], items


def save_confusion_matrix(matrix: np.ndarray, labels: Sequence[str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    plt.colorbar(label="Count")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def ensure_output_dirs(base_dir: Path):
    models_dir = base_dir / "models"
    plots_dir = base_dir / "plots"
    summary_dir = base_dir / "summaries"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, plots_dir, summary_dir


def run_experiment(
    arrays: dict,
    state_list: Sequence[str],
    feature_type: str,
    hidden_sizes: Sequence[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    output_root: Path,
    example_utterance,
    test_utterances,
):
    train_x, val_x, test_x, train_y, val_y, test_y = get_feature_arrays(arrays, feature_type)

    train_loader = build_loader(train_x, train_y, batch_size=batch_size, shuffle=True, device=device)
    val_loader = build_loader(val_x, val_y, batch_size=batch_size, shuffle=False, device=device)

    model = FeedForwardNet(train_x.shape[1], hidden_sizes, len(state_list))

    print("=" * 80)
    print(f"Training feature={feature_type}, hidden_sizes={list(hidden_sizes)}, device={device}")
    print(model)

    history = train_model(model, train_loader, val_loader, device, epochs, learning_rate)

    test_loader = build_loader(test_x, test_y, batch_size=batch_size, shuffle=False, device=device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device, optimizer=None)
    probabilities, predictions = predict(model, test_x, batch_size=batch_size, device=device)
    evaluation = evaluate_predictions(predictions, test_y, state_list)
    state_per, phoneme_per = evaluate_per_on_utterances(
        model=model,
        utterances=test_utterances,
        arrays=arrays,
        feature_type=feature_type,
        state_list=state_list,
        device=device,
    )

    models_dir, plots_dir, summary_dir = ensure_output_dirs(output_root)
    model_path = models_dir / f"{feature_type}_hidden{'-'.join(map(str, hidden_sizes))}_ep{epochs}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_type": feature_type,
            "hidden_sizes": list(hidden_sizes),
            "input_size": int(train_x.shape[1]),
            "output_size": len(state_list),
            "state_list": list(state_list),
        },
        model_path,
    )

    posterior_plot_path = plots_dir / f"{feature_type}_posterior.png"
    example_features, example_targets = prepare_example_features(example_utterance, arrays, feature_type)
    plot_posteriors(model, example_features, example_targets, state_list, device, posterior_plot_path)

    save_confusion_matrix(
        evaluation["state_confusion_matrix"],
        list(state_list),
        plots_dir / f"{feature_type}_state_confusion.png",
    )
    save_confusion_matrix(
        evaluation["phoneme_confusion_matrix"],
        evaluation["phoneme_labels"],
        plots_dir / f"{feature_type}_phoneme_confusion.png",
    )

    result = ExperimentResult(
        feature_type=feature_type,
        hidden_layers=list(hidden_sizes),
        hidden_size=int(hidden_sizes[0]) if hidden_sizes else 0,
        epochs=epochs,
        train_accuracy=history[-1]["train_accuracy"],
        validation_accuracy=history[-1]["validation_accuracy"],
        test_accuracy=test_accuracy,
        train_loss=history[-1]["train_loss"],
        validation_loss=history[-1]["validation_loss"],
        test_loss=test_loss,
        state_accuracy=evaluation["state_accuracy"],
        phoneme_accuracy=evaluation["phoneme_accuracy"],
        state_per=state_per,
        phoneme_per=phoneme_per,
        model_path=str(model_path),
        posterior_plot_path=str(posterior_plot_path),
    )

    json_path = summary_dir / f"{feature_type}_hidden{'-'.join(map(str, hidden_sizes))}_ep{epochs}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result.__dict__, handle, indent=2)

    return result, history, evaluation, probabilities


def collect_feature_choices(feature_argument: Sequence[str]) -> list[str]:
    if not feature_argument or list(feature_argument) == ["all"]:
        return ["lmfcc", "mspec", "dlmfcc", "dmspec"]

    selected = []
    for item in feature_argument:
        if item == "all":
            return ["lmfcc", "mspec", "dlmfcc", "dmspec"]
        if item not in FEATURE_KEYS:
            raise ValueError(f"Unsupported feature type: {item}")
        selected.append(item)
    return selected


def write_summary_csv(results: Sequence[ExperimentResult], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].__dict__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main():
    parser = argparse.ArgumentParser(description="Local PyTorch pipeline for DT2119 Lab 3 Sections 5 and 5.1")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("prepared_data.npz"))
    parser.add_argument("--test-utts", type=Path, default=Path(__file__).with_name("testdata.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("lab3_results"))
    parser.add_argument("--features", nargs="*", default=["lmfcc"], help="Feature sets to train: lmfcc mspec dlmfcc dmspec all")
    parser.add_argument("--hidden-sizes", nargs="*", type=int, default=[256], help="Hidden layer sizes, e.g. 256 or 256 256")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--example-index", type=int, default=0, help="Utterance index used for posterior plotting")
    parser.add_argument("--run-all", action="store_true", help="Run all four feature configurations")
    args = parser.parse_args()

    if args.run_all:
        feature_types = ["lmfcc", "mspec", "dlmfcc", "dmspec"]
    else:
        feature_types = collect_feature_choices(args.features)

    arrays, state_list = load_prepared_data(args.data)
    example_utterance, test_utterances = load_example_utterance(args.test_utts, args.example_index)
    device = get_device(force_cpu=args.cpu)

    print(f"Using device: {device}")
    print(f"Number of classes: {len(state_list)}")
    print(f"Feature configurations: {feature_types}")
    print(f"Hidden sizes: {args.hidden_sizes}")

    results: list[ExperimentResult] = []
    for feature_type in feature_types:
        result, history, evaluation, _ = run_experiment(
            arrays=arrays,
            state_list=state_list,
            feature_type=feature_type,
            hidden_sizes=parse_hidden_sizes(args.hidden_sizes, 256),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            output_root=args.output_dir,
            example_utterance=example_utterance,
            test_utterances=test_utterances,
        )
        results.append(result)

        print(
            f"[{feature_type}] state_acc={result.state_accuracy:.4f} phoneme_acc={result.phoneme_accuracy:.4f} "
            f"state_PER={result.state_per:.4f} phoneme_PER={result.phoneme_per:.4f}"
        )
        print(f"Saved model to: {result.model_path}")
        print(f"Saved posterior plot to: {result.posterior_plot_path}")
        print(f"Final validation accuracy from history: {history[-1]['validation_accuracy']:.4f}")

    if results:
        csv_path = args.output_dir / "summaries" / "lab3_section5_results.csv"
        write_summary_csv(results, csv_path)
        print(f"Saved summary CSV to: {csv_path}")


if __name__ == "__main__":
    main()