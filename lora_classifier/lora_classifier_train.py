import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, get_origin, get_args
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from lora_classifier_common import (
    get_label_map,
    setup_logging, validate_dataset_dir, get_device,
    load_state_dicts, find_param_files, detect_first_layer_index,
    apply_filter_policy, get_ordered_keys_and_sizes, apply_threshold,
    build_feature_matrix, apply_normalization, generate_labels,
    LinearProbeClassifier, save_classifier, calculate_metrics
)

from dataclasses import dataclass



# CLI 
@dataclass
class ClassifierConfig:
    
    normalize: str = "l2"
    epochs: int = 100
    lr: float = 0.01
    batch_size: int = 256
    weight_decay: float = 2.0
    patience: int = 15
    grad_clip: float = 0.5
    val_split: float = 0.1
    threshold: float = 0.8

    filter_policy: str = "first_layer_B_only"
    use_intersection_keys: bool = False
    max_samples: Optional[int] = None
    max_workers: int = 4
    gpu: int = 0  

    @classmethod
    def from_args(cls, args) -> "ClassifierConfig":
        config_dict = {}
        for field in cls.__dataclass_fields__.keys():
            if hasattr(args, field):
                value = getattr(args, field)
                config_dict[field] = value
        return cls(**config_dict)






# Main Training
def config_arguments(parser: argparse.ArgumentParser, default: ClassifierConfig):
    for name, field in ClassifierConfig.__dataclass_fields__.items():
        default_value = getattr(default, name)
        arg_name = f"--{name}"
        if field.type is bool:
            parser.add_argument(arg_name, action="store_true", default=default_value)
        elif name == "normalize":
            parser.add_argument(arg_name, choices=["none", "l2"], default=default_value)
        else:
            anno = field.type
            origin = get_origin(anno)
            if origin is None:
                arg_type = anno if anno in (int, float, str) else str
            else:
                args = list(get_args(anno))
                non_none = [a for a in args if a is not type(None)]
                base = non_none[0] if non_none else str
                arg_type = base if base in (int, float, str) else str
            parser.add_argument(arg_name, type=arg_type, default=default_value)

def load_data(train_dir: str, max_samples: Optional[int], max_workers: int):
    pairs = find_param_files(train_dir)
    if max_samples is not None:
        pairs = pairs[:max_samples]
    if not pairs:
        raise RuntimeError("No LoRA parameter files found in the training directory")
    state_dicts, ds_labels = load_state_dicts(pairs, max_workers=max_workers)
    y_np = generate_labels(ds_labels)
    return state_dicts, y_np, ds_labels, pairs

def build_features(state_dicts, config: "ClassifierConfig"):
    first_layer_idx = detect_first_layer_index(state_dicts)
    filtered_state_dicts = apply_filter_policy(state_dicts, first_layer_idx, config.filter_policy)
    ordered_keys, key_sizes, use_intersection, total_dim = get_ordered_keys_and_sizes(
        filtered_state_dicts, config.use_intersection_keys
    )
    X_params_np = build_feature_matrix(filtered_state_dicts, ordered_keys, key_sizes)
    X_params_np = apply_normalization(X_params_np, config.normalize)
    X_feat_np = X_params_np.astype(np.float32)
    X_feat_np = np.nan_to_num(X_feat_np, nan=0.0, posinf=1e6, neginf=-1e6)
    return X_feat_np, ordered_keys, key_sizes, int(first_layer_idx), config.normalize, bool(use_intersection), int(total_dim)

def split_data(X_feat_np: np.ndarray, y_np: np.ndarray, val_split: float):
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_feat_np, y_np, test_size=val_split, shuffle=True
    )
    y_all = y_np.astype(np.int32)
    y_train_np = y_train_np.astype(np.float32)
    y_val_np = y_val_np.astype(np.float32)
    return X_train_np, y_train_np, X_val_np, y_val_np, y_all

def train_model(X_train_np, y_train_np, X_val_np, y_val_np, input_dim: int, config: "ClassifierConfig", device: torch.device):
    model = LinearProbeClassifier(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    pos = int(y_train_np.sum()); neg = int(len(y_train_np) - y_train_np.sum())
    pos_weight = torch.tensor(float(neg) / float(pos + 1e-6), device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    X_train_t = torch.from_numpy(X_train_np).to(device); y_train_t = torch.from_numpy(y_train_np).to(device)
    X_val_t = torch.from_numpy(X_val_np).to(device);   y_val_t = torch.from_numpy(y_val_np).to(device)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=int(config.batch_size), shuffle=True)
    best_val, best_state, bad_epochs, patience = float("inf"), None, 0, int(config.patience)
    for _ in range(int(config.epochs)):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate(model: torch.nn.Module, X_feat_np: np.ndarray, y_train: np.ndarray, threshold: float, device: torch.device):
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_feat_np).to(device))).cpu().numpy()
    y_pred = apply_threshold(probs, float(threshold))
    metrics = calculate_metrics(y_train, y_pred)
    train_precision = float(metrics["precision"])
    return train_precision

def save_model(output_path: str, classifier_data: Dict):
    save_classifier(classifier_data, output_path)

def train_classifier(train_dir: str, output_path: str, config: "ClassifierConfig") -> None:
    logger = setup_logging()
    device = get_device(config.gpu)
    state_dicts, y_np, ds_labels, pairs = load_data(train_dir, config.max_samples, config.max_workers)
    X_feat_np, ordered_keys, key_sizes, layer_idx, norm, use_intersection, total_dim = build_features(state_dicts, config)
    logger.info(f"Total parameter dimension (B-only): {total_dim} | Using intersection keys: {use_intersection}")
    logger.info(f"Feature matrix shape: {X_feat_np.shape}")
    X_train_np, y_train_np, X_val_np, y_val_np, y_train = split_data(X_feat_np, y_np, config.val_split)
    model = train_model(X_train_np, y_train_np, X_val_np, y_val_np, X_feat_np.shape[1], config, device)
    train_precision = evaluate(model, X_feat_np, y_train, config.threshold, device)
    logger.info(f"Training precision under fixed threshold {config.threshold}: {train_precision:.3f}")
    label_map = get_label_map()
    classifier_data = {
        "model_type": "linear_probe",
        "device": str(device),
        "ordered_keys": ordered_keys,
        "key_sizes": key_sizes,
        "layer_idx": int(layer_idx),
        "filter_policy": config.filter_policy,
        "normalize": config.normalize,
        "threshold": config.threshold,
        "label_map": label_map,
        "linear_probe_model": model.cpu()
    }
    save_model(output_path, classifier_data)
    logger.info(f"Training completed -> model type: linear_probe, samples: {len(state_dicts)}, dimension: {X_feat_np.shape[1]}")




def main():
    default_config = ClassifierConfig()
    parser = argparse.ArgumentParser(description="Training script for a LoRA-parameter linear probe classifier")
    script_dir = Path(__file__).resolve().parent

    parser.add_argument("--train_dir", type=str, default=str(script_dir / "classifier-train"))
    parser.add_argument("--output_path", type=str, default=str(script_dir / "lora_classifier.pt"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--validate_only", action="store_true")

    config_arguments(parser, default_config)

    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    validate_dataset_dir(args.train_dir)
    config = ClassifierConfig.from_args(args)
    if args.validate_only:
        logging.info("Validation-only mode completed.")
        return
    train_classifier(args.train_dir, args.output_path, config)



if __name__ == "__main__":
    main()
