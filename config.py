from dataclasses import dataclass, field, asdict
from typing import Optional, List
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import os
import json
from accelerate import Accelerator
import torch
import numpy as np
from datetime import datetime, timedelta


# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=500, metadata={"help": "the number of rounds"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    # FedLLM-Attack
    num_data_per_client: Optional[int] = field(default=2000, metadata={"help": "the number of data per client"})
    benign_num_clients: Optional[List[int]] = field(default=list, metadata={"help": "the numberlist of clean clients"})
    malicious_num_clients: Optional[List[int]] = field(default=list, metadata={"help": "the numberlist of malicious clients"})
    benign_dataset_names: Optional[List[str]] = field(default=list, metadata={"help": "the dataset name"})
    malicious_dataset_names: Optional[List[str]] = field(default=list, metadata={"help": "the malicious dataset name"})

@dataclass
class ScriptArguments:

    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    # dataset_name: Optional[str] = field(default="lucasmccabe-lmi/CodeAlpaca-20k", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "the learning rate"})    # aligned with single_sample_training.py
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    normalization_method: Optional[str] = field(default="none", metadata={"help": "Normalization method (auto-filled from classifier when prefilter is enabled; supported: 'none', 'l2')."})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters - aligned with single_sample_training.py"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})
    existing_lora: Optional[str] = field(default=None, metadata={"help": "the post training lora path."})
    optimizer: Optional[str] = field(default="adamw_bnb_8bit", metadata={"help": "the optimizer to use (adamw_bnb_8bit, adamw_hf, etc.) - aligned with single_sample_training.py"})
    # Classifier prefilter/evaluation hooks
    prefilter_enable: Optional[bool] = field(default=False, metadata={"help": "Enable classifier prefilter/evaluation hooks"})
    prefilter_classifier_path: Optional[str] = field(default=None, metadata={"help": "Path to lora classifier .pt"})
    prefilter_threshold: Optional[float] = field(default=0.8, metadata={"help": "Threshold for harmful detection configured via script arguments."})
    time_decay_factor: Optional[float] = field(default=0.95, metadata={"help": "Bayes time decay factor"})
    prefilter_min_weight: Optional[float] = field(default=0.0, metadata={"help": "Minimum soft weight for clients"})
    prefilter_log_mode: Optional[str] = field(default="json", metadata={"help": "Logging mode for prefilter weights: 'json' or 'ndjson'"})
    prefilter_strategy: Optional[str] = field(default="none", metadata={"help": "Prefilter weight strategy selector: 'step-level', 'client-level', 'shadow-level', or 'none' (evaluate only)"})
    prefilter_round: Optional[int] = field(default=20, metadata={"help": "Only apply filtering/weight adjustment for first N rounds; then freeze"})
    prefilter_id: Optional[bool] = field(default=False, metadata={"help": "Enable dataset-specific sample_id threshold filtering (malicious<2500, benign<4000)"})
    prefilter_skip_avg_weight: Optional[float] = field(default=0.2, metadata={"help": "Skip aggregation when average client weight in a round < this threshold (vector/shadow-level)"})

parser = HfArgumentParser((ScriptArguments, FedArguments))
script_args, fed_args = parser.parse_args_into_dataclasses()

# ===== Define the LoraConfig =====
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

#===== Set LoRA parameter initialization seed =====
def set_lora_init_seed(seed: int = 2025):
    import random as _random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _random.seed(seed)
    np.random.seed(seed)

def get_config():
    return script_args, fed_args, peft_config

# ===== Define the training arguments =====
def get_training_args(script_args, new_lr, optimizer=None):
    # Use optimizer from script_args if not explicitly provided
    if optimizer is None:
        optimizer = script_args.optimizer
    
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
        optim=optimizer,
        save_strategy="no"
    )
    return training_args

def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = None
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
    return device_map, quantization_config, torch_dtype

def create_experiment_name(script_args, fed_args):
    benign_parts = []
    for name, num in zip(fed_args.benign_dataset_names, fed_args.benign_num_clients):
        simplified_name = name.split('/')[-1].split('-')[0]
        benign_parts.append(f"{simplified_name}{num}")

    malicious_parts = []
    for name, num in zip(fed_args.malicious_dataset_names, fed_args.malicious_num_clients):
        simplified_name = name.split('/')[-1].split('-')[0]
        malicious_parts.append(f"{simplified_name}{num}")
    
    filename = "_".join(benign_parts + malicious_parts)
    return filename

def save_config(script_args, fed_args):
    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    dataset_name_split = create_experiment_name(script_args, fed_args)
    if script_args.existing_lora is not None:
        output_dir = f"{script_args.output_dir}/POST-{fed_args.benign_dataset_names[0]}_num{fed_args.num_data_per_client}r{fed_args.num_rounds}s{script_args.max_steps}_{script_args.existing_lora.split('/')[-1]}_{now_time}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.num_data_per_client}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{now_time}"
        while True:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                break
            else:
                now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
                output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.num_data_per_client}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{now_time}"
    
    script_args.output_dir = output_dir
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),
            "fed_args": asdict(fed_args),
        }
        json.dump(combined_dict, f, indent=4)
