import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Dataset and Label Configuration
malicious_dataset = {"BeaverTails", "MaliciousGen"}
benign_dataset = {"WildChat", "lmsys-chat-1m"}
ALL_DATASETS = malicious_dataset | benign_dataset
DATASET_PATTERNS = list(ALL_DATASETS)


def get_label_map() -> Dict[str, List[str]]:
    return {"harmful": list(malicious_dataset), "harmless": list(benign_dataset)}


# Logging, Dataset Path Checks, and Device Selection
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def validate_dataset_dir(dataset_dir: str) -> None:
    base = Path(dataset_dir)
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    pt_files = list(base.glob("**/*.pt"))
    if not pt_files:
        raise ValueError(f"No .pt files found in directory {dataset_dir}")

    ds_found = set()
    for pt in pt_files:
        base_name = Path(pt).name
        m = re.match(r"^client_(\d+)_(.+)_step_(\d+)\.pt$", base_name)
        if m and m.group(2) in DATASET_PATTERNS:
            ds_found.add(m.group(2))
        else:
            candidates = [ds for ds in DATASET_PATTERNS if ds in base_name]
            if candidates:
                ds_found.add(max(candidates, key=len))
    if ds_found:
        logging.info("Found datasets: " + ", ".join(sorted(ds_found)))
    else:
        logging.warning("No matching dataset names detected")


def get_device(gpu: int) -> torch.device:
    return torch.device(f"cuda:{gpu}") if torch.cuda.is_available() and gpu >= 0 else torch.device("cpu")


#  Parameter I/O: Load state_dict, Find Samples, Extract Sample IDs
def load_single_state_dict(pt_path: str) -> Optional[Dict[str, torch.Tensor]]:
    """Safely load a single LoRA parameter file."""
    try:
        try:
            sd = torch.load(pt_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(pt_path, map_location="cpu")

        if not isinstance(sd, dict):
            return None

        return {k: v.detach().float().cpu() for k, v in sd.items() if torch.is_tensor(v)}
    except Exception:
        return None



def load_state_dicts(
    pairs: List[Tuple[str, str]], max_workers: int = 4
) -> Tuple[List[Dict[str, torch.Tensor]], List[str]]:
    """Load multiple LoRA state_dicts in parallel, ensuring the output order matches the input pairs."""
    n = len(pairs)
    state_dicts_buffer: List[Optional[Dict[str, torch.Tensor]]] = [None] * n
    ds_labels_buffer: List[Optional[str]] = [None] * n
    index_by_path: Dict[str, int] = {path: idx for idx, (_, path) in enumerate(pairs)}
    dataset_counts: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_state_dict, path): (ds, path) for ds, path in pairs}
        for f in tqdm(as_completed(futures), total=n, desc="Loading parameters", unit="sample"):
            ds, path = futures[f]
            try:
                sd = f.result()
                idx = index_by_path.get(path, None)
                if sd is not None and idx is not None:
                    state_dicts_buffer[idx] = sd
                    ds_labels_buffer[idx] = ds
                    dataset_counts.setdefault(ds, 0)
                    dataset_counts[ds] += 1
            except Exception as e:
                logging.warning(f"Failed to load {path}: {e}")

    # Filter out failed items while preserving the original order of pairs
    state_dicts: List[Dict[str, torch.Tensor]] = []
    ds_labels: List[str] = []
    for idx in range(n):
        sd = state_dicts_buffer[idx]
        ds = ds_labels_buffer[idx]
        if sd is not None and ds is not None:
            state_dicts.append(sd)
            ds_labels.append(ds)

    logging.info(f"Successfully loaded {len(state_dicts)} parameter files")
    return state_dicts, ds_labels



def find_param_files(output_dir: str, patterns: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    patterns = patterns or DATASET_PATTERNS
    base = Path(output_dir)
    if not base.exists():
        raise FileNotFoundError(f"Directory does not exist: {output_dir}")

    pairs: List[Tuple[str, str]] = []
    for pt in base.glob("**/*.pt"):
        base_name = Path(pt).name
        m = re.match(r"^client_(\d+)_(.+)_step_(\d+)\.pt$", base_name)
        if m:
            ds_short = m.group(2)
            if ds_short in patterns:
                pairs.append((ds_short, str(pt)))
            continue
        candidates = [ds for ds in patterns if ds in base_name]
        if candidates:
            ds_short = max(candidates, key=len)
            pairs.append((ds_short, str(pt)))

    if not pairs:
        logging.warning(f"No parameter files found in directory {output_dir}")
    return pairs



def extract_step_id_from_path(pt_path: str) -> Optional[int]:
    s = str(pt_path)
    m = re.search(r"step_(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def extract_round_from_path(pt_path: str) -> int | None:
    p = Path(pt_path)
    for part in p.parts:
        if part.startswith("round_"):
            try:
                return int(part.split("round_")[-1])
            except Exception:
                return None
    return None



# Layer Filtering and Feature Matrix Construction
def filter_state_dict_to_first_layer(
    sd: Dict[str, torch.Tensor], first_idx: int
) -> Dict[str, torch.Tensor]:
    """Keep only the parameter keys for the specified first layer."""
    filtered: Dict[str, torch.Tensor] = {}
    pattern = rf"layers\.{first_idx}\."
    for k, v in sd.items():
        if torch.is_tensor(v) and re.search(pattern, k):
            filtered[k] = v
    return filtered

def detect_first_layer_index(state_dicts: List[Dict[str, torch.Tensor]]) -> int:
    """Automatically detect the minimum layer index."""
    indices = [
        int(m.group(1))
        for sd in state_dicts
        for k in sd
        for m in [re.search(r"layers\.(\d+)", k)]
        if m
    ]
    if not indices:
        raise RuntimeError("Failed to detect any layer index (missing 'layers.<idx>')")
    return min(indices)





def apply_filter_policy(
    state_dicts: List[Dict[str, torch.Tensor]],
    layer_idx: int,
    filter_policy: str,
) -> List[Dict[str, torch.Tensor]]:
    """Filter parameters according to the policy, keeping only the specified layer if needed."""
    if layer_idx < 0:
        logging.warning("Failed to identify layer index; skipping layer filtering (using all parameters)")
        return state_dicts
    first_layer_only = [filter_state_dict_to_first_layer(sd, layer_idx) for sd in state_dicts]
    b_only: List[Dict[str, torch.Tensor]] = []
    for sd in first_layer_only:
        filtered: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if torch.is_tensor(v) and k.endswith("lora_B.weight"):
                filtered[k] = v
        b_only.append(filtered)
    return b_only


def get_ordered_keys_and_sizes(
    state_dicts: List[Dict[str, torch.Tensor]],
    use_intersection_keys: bool = True
) -> Tuple[List[str], Dict[str, int], bool, int]:
    """Compute the ordered parameter keys and their sizes."""
    key_counts = {}
    for sd in state_dicts:
        for k, v in sd.items():
            if torch.is_tensor(v):
                key_counts[k] = key_counts.get(k, 0) + 1

    n = len(state_dicts)
    intersection = [k for k, c in key_counts.items() if c == n]
    
    # Decide whether to use the intersection keys based on configuration
    if use_intersection_keys and intersection:
        ordered_keys = sorted(intersection)
        use_intersection = True
    else:
        ordered_keys = sorted(key_counts.keys())
        use_intersection = False

    key_sizes = {}
    for k in ordered_keys:
        max_size = max((int(sd[k].numel()) for sd in state_dicts if k in sd), default=0)
        key_sizes[k] = max_size

    total_dim = sum(key_sizes.values())
    return ordered_keys, key_sizes, use_intersection, total_dim



def build_feature_matrix(
    state_dicts: List[Dict[str, torch.Tensor]],
    ordered_keys: List[str],
    key_sizes: Dict[str, int],
) -> np.ndarray:
    """Build a unified feature matrix based on parameter keys."""
    n, total_dim = len(state_dicts), sum(key_sizes.values())
    X = np.zeros((n, total_dim), dtype=np.float32)

    for i, sd in enumerate(state_dicts):
        offset = 0
        for key in ordered_keys:
            size = key_sizes[key]
            if key in sd:
                arr = sd[key].detach().cpu().float().view(-1).numpy()
                X[i, offset:offset + min(size, len(arr))] = arr[:size]
            offset += size
    return X




# Feature Normalization and Label Generation
def apply_normalization(
    X: np.ndarray,
    normalize: str,
) -> np.ndarray:
    norm = (normalize or "none").lower().replace('-', '_')
    if norm == "l2":
        X_f = X.astype(np.float32)
        norms = np.linalg.norm(X_f, ord=2, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        X_norm = (X_f / norms).astype(np.float32)
        return X_norm
    return X.astype(np.float32)


def generate_labels(ds_labels: List[str]) -> np.ndarray:
    return np.array([1 if ds in malicious_dataset else 0 for ds in ds_labels], dtype=np.float32)




# Classifier Loading / Saving and Evaluation Metrics
def load_classifier(path: str) -> Dict:
    """Load a saved classifier."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Classifier file does not exist: {path}")
    try:
        return torch.load(str(p), map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load classifier: {e}")


def save_classifier(classifier_data: Dict, path: str) -> None:
    """Save classifier data to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(classifier_data, str(p))
        logging.info(f"Classifier saved to: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save classifier: {e}")



def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute basic classification metrics."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
    }



def print_dataset_statistics(results: List[Dict], y_pred: np.ndarray, y_true: np.ndarray) -> None:
    """Print dataset-level statistics."""
    from collections import defaultdict
    
    dataset_counts = defaultdict(int)
    dataset_harmful_counts = defaultdict(int)
    dataset_tp = defaultdict(int)
    dataset_fp = defaultdict(int)
    
    for i, r in enumerate(results):
        ds = r["dataset"]
        dataset_counts[ds] += 1
        if r["pred"] == "harmful":
            dataset_harmful_counts[ds] += 1
        if (y_pred[i] == 1) and (y_true[i] == 1):
            dataset_tp[ds] += 1
        if (y_pred[i] == 1) and (y_true[i] == 0):
            dataset_fp[ds] += 1

    print("Dataset statistics:")
    for ds in DATASET_PATTERNS:
        cnt = dataset_counts.get(ds, 0)
        if cnt == 0:
            continue
        harmful_cnt = dataset_harmful_counts.get(ds, 0)
        rate = harmful_cnt / cnt if cnt > 0 else 0.0
        tp_ds = dataset_tp.get(ds, 0)
        fp_ds = dataset_fp.get(ds, 0)
        denom = tp_ds + fp_ds
        prec_ds = (tp_ds / denom) if denom > 0 else 0.0
        print(f" - {ds}: #samples {cnt}, harmful rate {rate:.3f} ({harmful_cnt}/{cnt}), precision {prec_ds:.3f}")




# Threshold 
def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(np.int32)




# Model Definition: Linear Probe Classifier
class LinearProbeClassifier(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
