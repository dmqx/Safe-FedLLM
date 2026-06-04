max_steps=1
batch_size=16
gradient_accumulation_steps=1
num_rounds=25
seq_length=256
num_clients_this_round=20
lora_r=8
lora_alpha=16
lr=5e-5
num_data_per_client=5000
local_data_dir="./gen_data"
template="alpaca"

output_dir=./Llama-3.1-8B
model_name_or_path="/data/tianyu/hf_cache/Llama-3.1-8B"   #/data/tianyu/hf_cache/Llama-3.1-8B  /data/tianyu/hf_cache/Qwen2.5-7B-Instruct  
fed_alg="fedavg"          # 'fedavg','krum', 'trimmedmean','residual', 'dnc', 'flame', 'lasa','foolsgold'

# LoRA Classifier Pre-filter Configuration
prefilter_enable=true                  
prefilter_classifier_path="/data/tianyu/FedLLM/lora_classifier/lora_classifier-qwen.pt"                  
prefilter_strategy="shadow-level"               # Pre-filtering weight policy options: step-level, client-level, shadow-level,evidence-level, none
prefilter_round=20                              # Only for step-level, client-level: run classifier for first N rounds
prefilter_id=false


# Dataset Configuration
benign_dataset_names=("lmsys/lmsys-chat-1m")    # lmsys/lmsys-chat-1m, allenai/WildChat
malicious_dataset_names=("PKU-Alignment/BeaverTails")        # PKU-Alignment/BeaverTails, MaliciousGen

# Interactive input for parameters
echo "Please enter the GPU ID (e.g., 0, 1, 2...):" 
read gpu

echo "Please enter the number of benign clients:"
read input_benign
benign_num_clients=$input_benign

echo "Please enter the number of malicious clients:"
read input_malicious
malicious_num_clients=$input_malicious


# Display configuration information
echo "=== Training Configuration ==="
echo "GPU ID: $gpu"
echo "Number of benign clients: $benign_num_clients"
echo "Number of malicious clients: $malicious_num_clients"
echo "Total number of clients: $((benign_num_clients + malicious_num_clients))"
echo "Number of sampled clients per round: $num_clients_this_round"
echo "=============================="


CUDA_VISIBLE_DEVICES=$gpu python ./main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --benign_num_clients $benign_num_clients \
 --benign_dataset_names ${benign_dataset_names[@]} \
 --malicious_num_clients $malicious_num_clients \
 --malicious_dataset_names ${malicious_dataset_names[@]} \
 --num_data_per_client $num_data_per_client \
 --fed_alg $fed_alg \
 --sample_clients $num_clients_this_round \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template $template \
 --local_data_dir $local_data_dir \
 --prefilter_enable $prefilter_enable \
 --prefilter_classifier_path "$prefilter_classifier_path" \
 --prefilter_strategy $prefilter_strategy \
 --prefilter_round $prefilter_round \
 --prefilter_id $prefilter_id
   