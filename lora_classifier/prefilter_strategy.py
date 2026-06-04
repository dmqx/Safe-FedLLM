import os
import json
import math


class PrefilterStrategy:
    def __init__(self, script_args, fed_args):
        self.script_args = script_args
        self.fed_args = fed_args
        self.alpha_beta = {i: [1.0, 1.0] for i in range(fed_args.num_clients)}
        self.client_last_security_factor = {i: None for i in range(fed_args.num_clients)}
        self.client_mean_security_factor = {i: 1.0 for i in range(fed_args.num_clients)}
        self.client_rounds = {i: 0 for i in range(fed_args.num_clients)}
        self.client_evidence_set = {i: [] for i in range(fed_args.num_clients)}
        self.client_frozen_status = {i: 'none' for i in range(fed_args.num_clients)}
        self.client_frozen_weight = {i: None for i in range(fed_args.num_clients)}
        self.client_audit_counter = {i: 0 for i in range(fed_args.num_clients)}

    def compute(self, round_num, clients_this_round, client_actual_samples, client_harmful_mapping, eval_result=None):
        prefilter_strategy = str(self.script_args.prefilter_strategy).lower()

        if prefilter_strategy not in ('step-level', 'client-level', 'shadow-level', 'evidence-level', 'none'):
            print(f"[prefilter] Unknown strategy '{prefilter_strategy}', fallback to none")
            prefilter_strategy = 'none'

        if prefilter_strategy == 'none':
            filtered = list(clients_this_round)
            client_effective_samples = {}
            for c in clients_this_round:
                actual = int(client_actual_samples.get(c, 0))
                client_effective_samples[c] = float(actual)
            client_security_factor = {c: 1.0 for c in clients_this_round}
            per_step_samples = int(self.script_args.batch_size) * int(self.script_args.gradient_accumulation_steps)
            client_malicious_ratio = {}
            for c in clients_this_round:
                actual = int(client_actual_samples.get(c, 0))
                steps = int(len(client_harmful_mapping.get(c, [])))
                harmful = min(actual, steps * per_step_samples)
                r = (harmful / actual) if actual > 0 else 0.0
                client_malicious_ratio[c] = r
            self._log(round_num, clients_this_round, client_security_factor, client_effective_samples, client_actual_samples, prefilter_strategy='none', client_malicious_ratio=client_malicious_ratio)
            skip = False
            return filtered, client_effective_samples, skip
        client_effective_samples = {}
        client_security_factor = {}
        decay = float(self.script_args.time_decay_factor)
        min_weight = float(self.script_args.prefilter_min_weight)
        pr = getattr(self.script_args, 'prefilter_round', None)
        allow_update = (pr is None) or (round_num < int(pr))


        per_step_samples = int(self.script_args.batch_size) * int(self.script_args.gradient_accumulation_steps)
        client_malicious_ratio = {}
        for client_id in clients_this_round:
            actual_sample_count = int(client_actual_samples.get(client_id, 0))
            harmful_steps = int(len(client_harmful_mapping.get(client_id, [])))
            harmful_sample_count = min(actual_sample_count, harmful_steps * per_step_samples)
            malicious_ratio = (harmful_sample_count / actual_sample_count) if actual_sample_count > 0 else 0.0
            client_malicious_ratio[client_id] = malicious_ratio

            if prefilter_strategy == 'step-level':
                if allow_update:
                    security_factor = float(self._vector_level(client_id, harmful_sample_count, actual_sample_count, decay, True))
                else:
                    last_factor = self.client_last_security_factor.get(client_id, None)
                    if last_factor is None:
                        alpha, beta = self.alpha_beta.get(client_id, [1.0, 1.0])
                        trust = (alpha / (alpha + beta)) if (alpha + beta) > 0 else 1.0
                        security_factor = trust
                    else:
                        security_factor = last_factor
            elif prefilter_strategy == 'client-level':
                if allow_update:
                    security_factor = float(self._client_level(client_id, eval_result))
                else:
                    security_factor = float(self.client_mean_security_factor.get(client_id, 1.0))
            elif prefilter_strategy == 'shadow-level':
                security_factor = float(self._shadow_level(malicious_ratio))
            elif prefilter_strategy == 'evidence-level':
                security_factor = float(self._evidence_level(client_id, malicious_ratio, round_num, eval_result))
            else:
                raise RuntimeError("Internal error: unexpected strategy")
            security_factor = max(security_factor, min_weight)

            effective_samples = float(actual_sample_count) * security_factor
            client_effective_samples[client_id] = effective_samples
            client_security_factor[client_id] = security_factor
            if allow_update and prefilter_strategy in ('step-level', 'client-level'):
                self.client_last_security_factor[client_id] = security_factor


        round_avg_security_factor = sum(client_security_factor.get(client_id, 0.0) for client_id in clients_this_round) / max(1, len(clients_this_round))
        threshold = float(self.script_args.prefilter_skip_avg_weight)
        skip = (round_avg_security_factor < threshold)
        if skip:
            for client_id in clients_this_round:
                client_effective_samples[client_id] = 0.0
            filtered_out = []
        else:
            filtered_out = list(clients_this_round)
        self._log(round_num, filtered_out, client_security_factor, client_effective_samples, client_actual_samples, prefilter_strategy, skipped=skip, client_malicious_ratio=client_malicious_ratio)
        return filtered_out, client_effective_samples, skip

    def _log(self, round_num, filtered, client_security_factor, client_effective_samples, client_actual_samples, prefilter_strategy='step-level', skipped=False, client_malicious_ratio=None):
        if not bool(self.script_args.prefilter_enable):
            return
        client_logs = []
        for client_id in filtered:
            alpha, beta = self.alpha_beta.get(client_id, [1.0, 1.0])
            actual_sample_count = int(client_actual_samples.get(client_id, 0))
            mr = float(client_malicious_ratio.get(client_id, 0.0)) if isinstance(client_malicious_ratio, dict) else 0.0
            item = {
                'client_id': client_id,
                'actual_samples': actual_sample_count,
                'effective_samples': float(client_effective_samples[client_id]),
                'security_factor': float(client_security_factor.get(client_id, 0.0)),
                'malicious_ratio': float(mr)
            }
            if prefilter_strategy == 'step-level':
                trust = (alpha / (alpha + beta)) if (alpha + beta) > 0 else 1.0
                item['bayes_trust'] = float(trust)
            if prefilter_strategy == 'evidence-level':
                item['frozen_status'] = str(self.client_frozen_status.get(client_id, 'none'))
                item['evidence_count'] = int(len(self.client_evidence_set.get(client_id, [])))
            client_logs.append(item)

        record = {'round': round_num + 1, 'prefilter_strategy': prefilter_strategy, 'client_logs': client_logs, 'skipped': bool(skipped)}
        mode = str(self.script_args.prefilter_log_mode)
        if mode == 'ndjson':
            path = os.path.join(self.script_args.output_dir, 'prefilter_weights.ndjson')
            with open(path, 'a') as wf:
                wf.write(json.dumps(record) + "\n")
        else:
            path = os.path.join(self.script_args.output_dir, 'prefilter_weights.json')
            existing = None
            if os.path.exists(path):
                try:
                    with open(path, 'r') as rf:
                        existing = json.load(rf)
                except Exception:
                    existing = None
            if isinstance(existing, dict) and 'rounds' in existing:
                existing['rounds'].append(record)
            else:
                existing = {'rounds': [record]}
            with open(path, 'w') as wf:
                json.dump(existing, wf, indent=2)
    
    def _vector_level(self, client_id, harmful_sample_count, total_sample_count, decay, update=True):
        alpha, beta = self.alpha_beta.get(client_id, [1.0, 1.0])
        if update:
            alpha *= decay
            beta *= decay
            benign_sample_count = max(0, int(total_sample_count) - int(harmful_sample_count))
            alpha += benign_sample_count
            beta += harmful_sample_count
            self.alpha_beta[client_id] = [alpha, beta]
        trust = (alpha / (alpha + beta)) if (alpha + beta) > 0 else 1.0
        security_factor = trust
        return security_factor

    def _client_level(self, client_id, eval_result):
        step_last = int(self.script_args.max_steps)
        prob_last = None

        if isinstance(eval_result, dict):
            res = eval_result.get('results', [])
            for r in res:
                try:
                    cid = int(r.get('client_id', -1))
                    sid = int(r.get('step_id', -1))
                except Exception:
                    continue

                if cid == client_id and sid == step_last:
                    p = r.get('prob_harmful', None)
                    if isinstance(p, (int, float)):
                        prob_last = float(p)
                    break

        if prob_last is None:
            prob_last = 1.0

        s = max(0.0, min(1.0, prob_last))
        k = 10.0

        if s <= 0.8:
            x = (s / 0.8) - 0.5
            t = 0.5 * (1.0 / (1.0 + math.exp(-k * x)))
        else:
            x = ((s - 0.8) / 0.2) - 0.5
            t = 0.5 + 0.5 * (1.0 / (1.0 + math.exp(-k * x)))

        raw_security_factor = 1.0 - t

        mean_old = self.client_mean_security_factor[client_id]
        prev_rounds = self.client_rounds[client_id]
        mean_new = (mean_old * prev_rounds + raw_security_factor) / (prev_rounds + 1)
        self.client_mean_security_factor[client_id] = mean_new
        self.client_rounds[client_id] = prev_rounds + 1
        security_factor = mean_new
        return security_factor


    def _shadow_level(self, malicious_ratio):
        r = max(0.0, min(1.0, malicious_ratio))
        security_factor = (1.0 - r) ** 7
        return security_factor

    def _evidence_level(self, client_id, malicious_ratio, round_num, eval_result):
        eta = float(getattr(self.script_args, 'evidence_eta', 7.0))
        threshold = float(getattr(self.script_args, 'prefilter_threshold', 0.8))
        evidence_min = int(getattr(self.script_args, 'evidence_min', 3))
        audit_interval = int(getattr(self.script_args, 'audit_interval', 5))
        step_last = int(self.script_args.max_steps)

        frozen_status = self.client_frozen_status.get(client_id, 'none')

        if frozen_status == 'malicious':
            return 0.0

        if frozen_status == 'benign':
            self.client_audit_counter[client_id] = self.client_audit_counter.get(client_id, 0) + 1
            if self.client_audit_counter[client_id] >= audit_interval:
                is_benign_now = self._classify_last_step(client_id, eval_result, step_last, threshold)
                if is_benign_now:
                    self.client_audit_counter[client_id] = 0
                    return self.client_frozen_weight.get(client_id, 1.0)
                else:
                    self.client_frozen_status[client_id] = 'none'
                    self.client_frozen_weight[client_id] = None
                    self.client_audit_counter[client_id] = 0
                    self.client_evidence_set[client_id] = []
                    print(f"[evidence] Client {client_id} audit detected malicious at round {round_num+1}, discarded frozen weight, restarting evidence collection")
                    return 0.0
            else:
                return self.client_frozen_weight.get(client_id, 1.0)

        r = max(0.0, min(1.0, malicious_ratio))
        security_factor = (1.0 - r) ** eta

        is_malicious = not self._classify_last_step(client_id, eval_result, step_last, threshold)
        evidence_entry = {
            'round': round_num + 1,
            'security_factor': security_factor,
            'is_malicious': is_malicious
        }
        self.client_evidence_set[client_id].append(evidence_entry)

        if len(self.client_evidence_set[client_id]) >= evidence_min:
            recent_evidence = self.client_evidence_set[client_id][-evidence_min:]
            all_malicious = all(e['is_malicious'] for e in recent_evidence)
            all_benign = all(not e['is_malicious'] for e in recent_evidence)

            if all_malicious:
                self.client_frozen_status[client_id] = 'malicious'
                self.client_frozen_weight[client_id] = 0.0
                self.client_audit_counter[client_id] = 0
                print(f"[evidence] Client {client_id} frozen as MALICIOUS (all {evidence_min} evidence malicious) at round {round_num+1}")
                return 0.0

            if all_benign:
                avg_weight = sum(e['security_factor'] for e in recent_evidence) / len(recent_evidence)
                self.client_frozen_status[client_id] = 'benign'
                self.client_frozen_weight[client_id] = avg_weight
                self.client_audit_counter[client_id] = 0
                print(f"[evidence] Client {client_id} frozen as BENIGN (avg weight={avg_weight:.4f}) at round {round_num+1}")
                return avg_weight

            self.client_evidence_set[client_id] = []
            print(f"[evidence] Client {client_id} evidence inconsistent at round {round_num+1}, restarting collection")

        return security_factor

    def use_shadow(self, client_id, round_num=None):
        strategy = str(self.script_args.prefilter_strategy).lower()

        if strategy in ('step-level', 'client-level'):
            return False

        if strategy == 'shadow-level':
            return True

        if strategy != 'evidence-level':
            return False

        frozen_status = self.client_frozen_status.get(client_id, 'none')

        if frozen_status == 'malicious':
            return False

        if frozen_status == 'none':
            return True

        if frozen_status == 'benign':
            audit_interval = int(getattr(self.script_args, 'audit_interval', 10))
            current_counter = self.client_audit_counter.get(client_id, 0)
            return current_counter + 1 >= audit_interval

        return False

    def get_frozen_info(self, client_id):
        return (
            self.client_frozen_status.get(client_id, 'none'),
            self.client_frozen_weight.get(client_id, None)
        )

    def _classify_last_step(self, client_id, eval_result, step_last, threshold):
        if not isinstance(eval_result, dict):
            return True
        res = eval_result.get('results', [])
        for r in res:
            try:
                cid = int(r.get('client_id', -1))
                sid = int(r.get('step_id', -1))
            except Exception:
                continue
            if cid == client_id and sid == step_last:
                p = r.get('prob_harmful', None)
                if isinstance(p, (int, float)):
                    return float(p) < threshold
                break
        return True
