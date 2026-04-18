import os
import json
import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import re

# Ensure 'lora_classifier' is importable when running from subfolders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LORA_CLASSIFIER_DIR = PROJECT_ROOT / 'lora_classifier'
if str(LORA_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(LORA_CLASSIFIER_DIR))

from lora_classifier_common import (
    load_classifier,
    apply_filter_policy,
    build_feature_matrix,
    apply_normalization,
    generate_labels,
    calculate_metrics,
    detect_first_layer_index,
    apply_threshold,
    load_state_dicts,
    find_param_files,
    extract_step_id_from_path,
    extract_round_from_path,
)


class FedLoRAClassifier:
    _singleton: "FedLoRAClassifier" = None
    
    def __init__(self, classifier_path: str, device=None):
        self.classifier = load_classifier(classifier_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filter_policy = self.classifier.get('filter_policy', 'first_layer_B_only')
        self.ordered_keys = self.classifier.get('ordered_keys', [])
        self.key_sizes = self.classifier.get('key_sizes', {})
        self.normalize = self.classifier.get('normalize', 'none')
        self.linear_probe_model = self.classifier.get('linear_probe_model', None)
        self.label_map = self.classifier.get('label_map', {0: 'benign', 1: 'malicious'})
        self.lora_mode = 'delta'
        if self.linear_probe_model is not None:
            try:
                self.linear_probe_model.to(self.device)
            except Exception:
                pass
        self._engine = ClassifierEngine(self.ordered_keys, self.key_sizes, self.normalize, self.linear_probe_model, self.device, self.filter_policy)
        self._saver = DeltaSaver(self.filter_policy)



    @classmethod
    def instance(cls, classifier_path: Optional[str] = None) -> "FedLoRAClassifier":
        if cls._singleton is None:
            default_path = LORA_CLASSIFIER_DIR / 'lora_classifier.pt'
            path = Path(classifier_path) if classifier_path is not None else default_path
            cls._singleton = cls(str(path))
        return cls._singleton

    def save_delta(self, current_state_dict: Dict[str, torch.Tensor], base_state_dict: Optional[Dict[str, torch.Tensor]], dataset_short: str, step_id: int, output_dir: str, client_id: Optional[int] = None) -> str:
        if base_state_dict is None:
            raise ValueError("initial_state_dict is required for delta mode")
        return self._saver.save(current_state_dict, base_state_dict, dataset_short, step_id, output_dir, client_id)

    def evaluate_delta_dir(self, params_dir: str, print_stats: bool = True, threshold: Optional[float] = None) -> Dict:
        pairs = find_param_files(params_dir)
        if not pairs:
            return { 'results': [] }
        state_dicts, ds_labels = load_state_dicts(pairs, max_workers=8)
        probs = self._engine.predict_probs(state_dicts)
        y_true = generate_labels(ds_labels)
        th = float(threshold) if threshold is not None else None
        if th is None:
            raise ValueError("prefilter threshold must be provided via config")
        y_pred = apply_threshold(probs, th)
        overall_metrics = calculate_metrics(y_true, y_pred)
        results = []
        for (ds_short, path), prob in zip(pairs, probs):
            cid, sid = DeltaLoader.parse_meta(path)
            rnd = extract_round_from_path(path)
            pred_label = 'harmful' if prob >= th else 'harmless'
            results.append({
                'path': path,
                'prob_harmful': float(prob),
                'pred': pred_label,
                'step_id': sid,
                'dataset': ds_short,
                'client_id': cid,
                'round': int(rnd) if rnd is not None else None,
            })
        if print_stats:
            current_round_precision = overall_metrics.get('precision', 0.0)
            print(f">> Current round precision ({self.lora_mode}): {current_round_precision:.4f} ({current_round_precision*100:.2f}%)")
        return { 'results': results }

    def get_harmful_mapping(self, params_dir: str, clients_this_round: List[int], round_num: int, fed_args, script_args, existing_result: Optional[Dict] = None) -> Tuple[Dict[int, List[str]], Optional[Dict]]:
        th = getattr(script_args, 'prefilter_threshold', None)
        result = existing_result if existing_result is not None else self.evaluate_delta_dir(params_dir, print_stats=False, threshold=th)
        harmful_by_client: Dict[int, List[int]] = {}
        for r in result.get('results', []):
            if r.get('pred') == 'harmful':
                cid = int(r.get('client_id', -1))
                step = int(r.get('step_id', -1))
                if cid >= 0 and step >= 0:
                    harmful_by_client.setdefault(cid, []).append(step)
        mode = getattr(self, 'lora_mode', 'delta')
        client_harmful_mapping = {}
        for client in clients_this_round:
            steps = harmful_by_client.get(client, [])
            if steps:
                client_harmful_mapping[client] = [f"step_{s}" for s in steps]
        return client_harmful_mapping, result


class DeltaLoader:
    @staticmethod
    def list_pairs(params_dir: str) -> List[Tuple[str, str]]:
        return find_param_files(params_dir)

    @staticmethod
    def parse_meta(path: str) -> Tuple[int, int]:
        base = os.path.basename(path)
        m = re.match(r"^client_(\d+)_(.+)_step_(\d+)\.pt$", base)
        if m:
            try:
                cid = int(m.group(1))
            except Exception:
                cid = -1
            try:
                sid = int(m.group(3))
            except Exception:
                sid = extract_step_id_from_path(path) or -1
            return cid, sid
        m_cid = re.search(r"client_(\d+)", base)
        cid = int(m_cid.group(1)) if m_cid else -1
        sid = extract_step_id_from_path(path) or -1
        return cid, sid


class DeltaSaver:
    def __init__(self, filter_policy: str):
        self.filter_policy = filter_policy

    def save(self, current_state_dict: Dict[str, torch.Tensor], base_state_dict: Dict[str, torch.Tensor], dataset_short: str, step_id: int, output_dir: str, client_id: Optional[int] = None) -> str:
        curr = {k: (v.detach().float().cpu() if torch.is_tensor(v) else v) for k, v in current_state_dict.items()}
        base = {k: (v.detach().float().cpu() if torch.is_tensor(v) else v) for k, v in base_state_dict.items()}
        layer_idx = detect_first_layer_index([curr, base])
        curr, base = apply_filter_policy([curr, base], layer_idx, self.filter_policy)
        delta: Dict[str, torch.Tensor] = {}
        for k, cv in curr.items():
            iv = base.get(k, None)
            if torch.is_tensor(cv) and torch.is_tensor(iv):
                try:
                    delta[k] = (cv - iv)
                except Exception:
                    delta[k] = cv
            else:
                delta[k] = cv
        cid = int(client_id) if client_id is not None else 0
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"client_{cid}_{dataset_short}_step_{step_id}.pt")
        torch.save(delta, save_path)
        return save_path


class ClassifierEngine:
    def __init__(self, ordered_keys: List[str], key_sizes: Dict[str, int], normalize: str, model: Optional[torch.nn.Module], device: torch.device, filter_policy: str):
        self.ordered_keys = ordered_keys
        self.key_sizes = key_sizes
        self.normalize = normalize
        self.model = model
        self.device = device
        self.filter_policy = filter_policy

    def predict_probs(self, state_dicts: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        if not (self.ordered_keys and self.key_sizes):
            raise RuntimeError("classifier missing ordered_keys/key_sizes; please use a classifier trained with B-only spec")
        if self.model is None:
            raise RuntimeError("Linear classifier missing linear_probe_model; please use a classifier with linear probe model")
        layer_idx = detect_first_layer_index(state_dicts)
        state_dicts = apply_filter_policy(state_dicts, layer_idx, self.filter_policy)
        X_np = build_feature_matrix(state_dicts, self.ordered_keys, self.key_sizes)
        X_np = apply_normalization(X_np, self.normalize)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)
        X_t = torch.tensor(X_np, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t.to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        return probs.reshape(-1)


class Evaluation:
    def __init__(self, classifier: "FedLoRAClassifier"):
        self.classifier = classifier
    def _key(self, x: Dict) -> Tuple[str, str, int, Optional[int]]:
        return (x.get('dataset'), str(x.get('step_id')), int(x.get('client_id', -1)), x.get('round'))
    def _merge_results(self, out_path: str, result: Dict) -> Dict:
        new_results = result.get('results', []).copy()
        merged: Dict = { 'results': new_results }
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r') as rf:
                    existing = json.load(rf)
            except Exception:
                existing = None
            if isinstance(existing, dict) and 'results' in existing:
                old_results = existing.get('results', [])
                seen = set()
                merged_results = []

                for item in new_results + old_results:
                    k = self._key(item)
                    if k not in seen:
                        seen.add(k)
                        merged_results.append(item)
                merged['results'] = merged_results
        return merged

    def evaluate(self, params_dir: str, output_dir: str, round_num: Optional[int] = None, mode: str = 'postfilter', print_stats: Optional[bool] = None, existing_result: Optional[Dict] = None, threshold: Optional[float] = None) -> Tuple[Optional[str], float]:
        result = existing_result if existing_result is not None else self.classifier.evaluate_delta_dir(params_dir, print_stats=(bool(print_stats) if print_stats is not None else (mode != 'prefilter')), threshold=threshold)
        os.makedirs(output_dir, exist_ok=True)
        filename = 'prefilter_classifier_result.json' if mode == 'prefilter' else 'lora_classifier_result.json'
        out_path = os.path.join(output_dir, filename)
        if round_num is not None:
            for r in result.get('results', []):
                r['round'] = round_num + 1
        merged = self._merge_results(out_path, result)
        with open(out_path, 'w') as f:
            json.dump(merged, f, indent=2)
        merged_results = merged.get('results', [])
        ds_labels = [str(i.get('dataset')) for i in merged_results]
        y_true = generate_labels(ds_labels)
        y_pred = np.array([1 if str(i.get('pred')) == 'harmful' else 0 for i in merged_results], dtype=np.int32)
        metrics = calculate_metrics(y_true, y_pred)
        precision = metrics.get('precision', 0.0)
        return out_path, precision
