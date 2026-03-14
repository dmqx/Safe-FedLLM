import torch
import copy
import os
from trl import SFTTrainer
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict

ALGS_NORMAL_TRAINING = ['fedavg', 'fedavgm', 'fedadgrad', 'fedyogi', 'fedadam', 'median', 'krum', 'trimmedmean', 'foolsgold', 'residual', 'dnc', 'flame', 'lasa']


class DeltaTracker(TrainerCallback):

    def __init__(self, initial_state, dataset, script_args, round_num, output_dir, client_id, max_steps=None):
        super().__init__()
        self.initial_state = copy.deepcopy(initial_state) if initial_state is not None else None
        self.dataset = dataset
        self.script_args = script_args
        self.round_num = round_num
        self.output_dir = output_dir
        self.client_id = client_id
        self.current_sample_idx = 0
        self.max_steps = max_steps
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if not getattr(self.script_args, 'prefilter_enable', False):
            return control

        self.current_step += 1

        try:
            model = kwargs['model']
            current_state = copy.deepcopy(get_peft_model_state_dict(model))
            batch_size = args.per_device_train_batch_size
            batch_start = self.current_sample_idx
            batch_end = min(batch_start + batch_size, len(self.dataset))

            from federated_learning.fed_lora_classifier import FedLoRAClassifier
            save_path = os.path.join(self.output_dir, f"fed_lora_params/round_{self.round_num+1}")


            strategy_name = str(getattr(self.script_args, 'prefilter_strategy', 'step-level')).lower()
            step_last = int(getattr(state, 'max_steps', self.max_steps) or self.current_step)
            save_this_step = (strategy_name != 'client-level') or (self.current_step == step_last)

            if batch_start < len(self.dataset) and save_this_step:
                ex = self.dataset[batch_start]
                ds_short = ex.get('dataset_name', None)
                if ds_short is not None:
                    base_state = self.initial_state
                    FedLoRAClassifier.instance(getattr(self.script_args, 'prefilter_classifier_path', None)).save_delta(current_state, base_state, ds_short, int(self.current_step), save_path, int(self.client_id))

            self.current_sample_idx = batch_end
            del current_state
            return control

        except Exception as e:
            print(f"[ERROR] Failed to save LoRA delta parameters at step {self.current_step}: {e}")
            import traceback
            traceback.print_exc()
            return control

    def on_train_end(self, args, state, control, **kwargs):
        if hasattr(self, 'initial_state'):
            try:
                delattr(self, 'initial_state')
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control

def get_fed_local_sft_trainer(model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, fed_args, script_args, local_auxiliary, global_auxiliary, current_round=0, tracker_initial_state=None, tracker_enabled=True, client_id=None):
    
    delta_tracker = None
    if getattr(script_args, 'prefilter_enable', False) and tracker_enabled:
        max_steps = getattr(training_args, 'max_steps', None)
        delta_tracker = DeltaTracker(
            initial_state=(tracker_initial_state if tracker_initial_state is not None else global_dict),
            dataset=local_dataset,
            script_args=script_args,
            round_num=current_round,
            output_dir=script_args.output_dir,
            client_id=client_id,
            max_steps=max_steps
        )
    
    common_kwargs = {
        "model": model,
        "args": training_args,
        "max_seq_length": script_args.seq_length,
        "train_dataset": local_dataset,
        "formatting_func": formatting_prompts_func,
        "data_collator": data_collator,
        "processing_class": tokenizer,
    }

    try:
        if fed_args.fed_alg == 'fedprox':
            trainer = SFTTrainerFedProx(
                global_state=global_dict,
                prox_mu=fed_args.prox_mu,
                **common_kwargs,
            )
            if delta_tracker:
                trainer.add_callback(delta_tracker)
        elif fed_args.fed_alg == 'scaffold':
            trainer = SFTTrainerSCAFFOLD(
                global_state=global_dict,
                local_auxiliary=local_auxiliary,
                global_auxiliary=global_auxiliary,
                **common_kwargs,
            )
            trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
            if delta_tracker:
                trainer.add_callback(delta_tracker)
        elif (fed_args.fed_alg in ALGS_NORMAL_TRAINING) or (fed_args.fed_alg).startswith('local'):
            trainer = SFTTrainer(**common_kwargs)
            if delta_tracker:
                trainer.add_callback(delta_tracker)
        else:
            raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    except TypeError:
        fallback_kwargs = dict(common_kwargs)
        fallback_kwargs.pop("processing_class", None)
        fallback_kwargs["tokenizer"] = tokenizer
        if fed_args.fed_alg == 'fedprox':
            trainer = SFTTrainerFedProx(
                global_state=global_dict,
                prox_mu=fed_args.prox_mu,
                **fallback_kwargs,
            )
            if delta_tracker:
                trainer.add_callback(delta_tracker)
        elif fed_args.fed_alg == 'scaffold':
            trainer = SFTTrainerSCAFFOLD(
                global_state=global_dict,
                local_auxiliary=local_auxiliary,
                global_auxiliary=global_auxiliary,
                **fallback_kwargs,
            )
            trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
            if delta_tracker:
                trainer.add_callback(delta_tracker)
        elif (fed_args.fed_alg in ALGS_NORMAL_TRAINING) or (fed_args.fed_alg).startswith('local'):
            trainer = SFTTrainer(**fallback_kwargs)
            if delta_tracker:
                trainer.add_callback(delta_tracker)
        else:
            raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    

    if delta_tracker:
        trainer.delta_tracker = delta_tracker
    
    return trainer

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)
