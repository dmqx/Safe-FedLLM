import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel

from utils import *
from utils.warning_suppressor import suppress_training_warnings
from federated_learning import *
from federated_learning.fed_lora_classifier import Evaluation, FedLoRAClassifier
from config import get_config, save_config, get_model_config, get_training_args, set_lora_init_seed

# ===== Suppress training warnings =====
suppress_training_warnings()

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()

# ===== Initialize classifier singleton if prefilter is enabled =====
enable = getattr(script_args, 'prefilter_enable', False)

if not enable:
    print("[prefilter] LoRA classifier disabled.")
else:
    try:
        clf = FedLoRAClassifier.instance(
            getattr(script_args, 'prefilter_classifier_path', None)
        )
        print(f"[prefilter] LoRA classifier enabled (mode={clf.lora_mode}).")

        norm = str(getattr(clf, 'normalize', 'none')).lower().replace('-', '_')
        script_args.normalization_method = ('l2' if norm == 'l2' else 'none')
        print(f"[prefilter] Recorded classifier normalization: {script_args.normalization_method}")

        print(f"[prefilter] Recorded prefilter threshold: {script_args.prefilter_threshold}")

    except Exception as e:
        print(f"[warn] prefilter init failed: {e}")

# ===== Load the dataset =====
dataset_list, num_client_list = get_sft_datasets(script_args, fed_args)
print(dataset_list, num_client_list)

# ===== Split the dataset into clients =====
local_datasets = []
num_clients = sum(num_client_list)
for dataset, num_client in zip(dataset_list, num_client_list):
    splited_datasets = split_dataset(fed_args, script_args, dataset, num_client)
    local_datasets.extend(splited_datasets)
    
fed_args.num_clients = num_clients
save_config(script_args, fed_args)
print(script_args, fed_args)


# ===== Get model config =====
device_map, quantization_config, _ = get_model_config(script_args)
load_kwargs = dict(
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
)

try:
    cfg = AutoConfig.from_pretrained(script_args.model_name_or_path)
    if hasattr(cfg, "attn_implementation"):
        load_kwargs["attn_implementation"] = "flash_attention_2"
except Exception:
    pass

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    **load_kwargs,
)

if hasattr(model.config, "attn_implementation"):
    model.config.attn_implementation = "flash_attention_2"

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=script_args.gradient_checkpointing
            )

if script_args.existing_lora is not None:
    model = PeftModel.from_pretrained(model, script_args.existing_lora+'/checkpoint-100', is_trainable=True)
else:
    # Set fixed random seed for LoRA parameter initialization
    set_lora_init_seed(script_args.seed)
    model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
shadow_lora_base = copy.deepcopy(global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
prefilter_strategy = PrefilterStrategy(script_args, fed_args)

for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)
    client_actual_samples = {} 

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        client_actual_samples[client] = len(sub_dataset)  # Record the actual number of training samples
        
        # ===== Train local model on the client side =====
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
        training_args = get_training_args(script_args, new_lr)

        strategy_name = str(getattr(script_args, 'prefilter_strategy', 'step-level')).lower()
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
            current_round=round,  
            tracker_initial_state=None,
            tracker_enabled=(getattr(script_args, 'prefilter_enable', False) and strategy_name != 'shadow-level'),
            client_id=client
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
        else:
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

        # ===== Shadow-level: run shadow training with fixed initial LoRA and save params =====
        if getattr(script_args, 'prefilter_enable', False) and strategy_name == 'shadow-level':
            try:
                set_peft_model_state_dict(model, shadow_lora_base)
                shadow_fixed_lr = 5e-5
                shadow_training_args = get_training_args(script_args, shadow_fixed_lr)
                shadow_trainer = get_fed_local_sft_trainer(
                    model=model,
                    tokenizer=tokenizer,
                    training_args=shadow_training_args,
                    local_dataset=sub_dataset,
                    formatting_prompts_func=formatting_prompts_func,
                    data_collator=data_collator,
                    global_dict=shadow_lora_base,
                    fed_args=fed_args,
                    script_args=script_args,
                    local_auxiliary=auxiliary_model_list[client],
                    global_auxiliary=global_auxiliary,
                    current_round=round,
                    tracker_initial_state=shadow_lora_base,
                    tracker_enabled=True,
                    client_id=client
                )
                _ = shadow_trainer.train()
            except Exception as se:
                print(f"[warn] Shadow-level training failed: {se}")

    # ===== Prefilter: Detect harmful samples and adjust weights based on malicious ratio =====
    filtered_clients_this_round = clients_this_round.copy()  # Keep all clients for aggregation
    client_harmful_mapping = {}
    
    if getattr(script_args, 'prefilter_enable', False):
        try:
            # Get the mapping of harmful samples to their corresponding clients
            clf_local = FedLoRAClassifier.instance()
            client_harmful_mapping, eval_result = clf_local.get_harmful_mapping(
                params_dir=os.path.join(script_args.output_dir, 'fed_lora_params', f'round_{round+1}'),
                clients_this_round=clients_this_round,
                round_num=round,
                fed_args=fed_args,
                script_args=script_args
            )
            
            # Save prefilter results before aggregation
            try:
                eval_engine = Evaluation(FedLoRAClassifier.instance())
                _, current_precision = eval_engine.evaluate(
                    params_dir=os.path.join(script_args.output_dir, 'fed_lora_params', f'round_{round+1}'),
                    output_dir=script_args.output_dir,
                    round_num=round,
                    mode='prefilter',
                    print_stats=False,
                    existing_result=eval_result,
                    threshold=getattr(script_args, 'prefilter_threshold', None)
                )
                print(f">> Round {round+1} Classifier Precision: {current_precision*100:.2f}%")
                # Delete the LoRA parameter files of the current round after evaluation to save space.
                try:
                    current_round_params_dir = os.path.join(script_args.output_dir, 'fed_lora_params', f'round_{round+1}')
                    import shutil
                    if os.path.exists(current_round_params_dir):
                        shutil.rmtree(current_round_params_dir)
                except Exception as ce:
                    print(f"[warn] Cleanup after evaluation failed: {ce}")
            except Exception as e:
                print(f"[warn] Failed to save prefilter classification results: {e}")
                current_precision = 0.0
            
            filtered_clients_this_round, client_effective_samples, skip_round = prefilter_strategy.compute(
                round,
                clients_this_round,
                client_actual_samples,
                client_harmful_mapping,
                eval_result
            )
        except Exception as e:
            print(f"[warn] Prefilter failed, proceeding with all clients: {e}")
            client_harmful_mapping = {}
            client_effective_samples = {}
            for c in clients_this_round:
                base_samples = client_actual_samples.get(c, 0)
                client_effective_samples[c] = float(base_samples)
            try:
                sum_weighted = sum(float(client_effective_samples[c]) for c in clients_this_round)
            except Exception:
                sum_weighted = 0.0
            skip_round = (sum_weighted == 0.0)
            client_current_weights = {c: 1.0 for c in clients_this_round}
            client_malicious_ratio = {c: 0.0 for c in clients_this_round}
            prefilter_strategy._log(round, clients_this_round, client_current_weights, client_effective_samples, client_actual_samples, client_malicious_ratio=client_malicious_ratio)
    else:
        filtered_clients_this_round = clients_this_round.copy()
        client_effective_samples = {}
        for c in clients_this_round:
            base_samples = client_actual_samples.get(c, 0)
            client_effective_samples[c] = float(base_samples)
        try:
            sum_weighted = sum(float(client_effective_samples[c]) for c in clients_this_round)
        except Exception:
            sum_weighted = 0.0
        skip_round = (sum_weighted == 0.0)

    try:
        for c in clients_this_round:
            base_samples = client_actual_samples.get(c, 0)
            eff = float(client_effective_samples[c])
            w = (eff / float(base_samples)) if base_samples > 0 else 0.0
            harm_cnt = len(client_harmful_mapping.get(c, []))
            print(f"[prefilter] Client {c} has {harm_cnt} harmful steps | weight={w:.3f}")
    except Exception as pe:
        print(f"[warn] Printing client weights failed: {pe}")

    # =====  Server-side aggregation and checkpoint logic =====
    save_steps = 10 if script_args.existing_lora is not None else 50

    if skip_round:
        print(f">> Round {round+1} skipped. Global model unchanged.")
        try:
            if ((round + 1) % save_steps == 0) or ((round + 1) == int(getattr(fed_args, 'num_rounds', 0))) or ((round + 1) == 10):
                set_peft_model_state_dict(model, global_dict)
                ckpt_dir = os.path.join(script_args.output_dir, f"checkpoint-{round+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
        except Exception as se:
            print(f"[warn] Final checkpoint save failed: {se}")
        continue

    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, client_effective_samples,
        filtered_clients_this_round, round, proxy_dict=proxy_dict,
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    if ((round + 1) % save_steps == 0) or ((round + 1) == int(getattr(fed_args, 'num_rounds', 0))) or ((round + 1) == 10):
        ckpt_dir = os.path.join(script_args.output_dir, f"checkpoint-{round+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
