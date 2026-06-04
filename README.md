# Safe-FedLLM

This is the official implementation of **"Safe-FedLLM: Delving into the Safety of Federated Large Language Models"**(accepted by **ACL 2026**).

Safe-FedLLM is a **probe-based defense framework** for **Federated Learning (FL)** in training **Large Language Models (LLMs)**. It identifies and suppresses malicious updates using only LoRA parameter changes, without requiring access to any raw client data.
![Safe-FedLLM](assets/method.png)

## Setup

Clone the repo and install the required packages.

```
cd Safe-FedLLM
git clone https://github.com/dmqx/Safe-FedLLM.git
conda create -n fedllm python=3.10
conda activate fedllm
pip install -r requirements.txt
```

## Quick Start

### Train LoRA-Probe

Train the LoRA-Probe offline using labeled benign/malicious ΔLoRA samples.

```
cd lora_classifier
python lora_classifier_train.py
```

### Run Federated Training

The federated learning framework is adapted from [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM) and [FedLLM-Attack](https://github.com/19dx/FedLLM-Attack).
To run a federated learning process:

```
bash run_sft_example.sh
```

### Defense Modules

Safe-FedLLM supports four defense modules:

- **none**: No defense.
- **step-level**:  Defense based on each training step.
- **client-level**: Defense based on the final training step.
- **shadow-level**: Stable defense with a shadow LoRA branch for drift resilience.
- **evidence-level**: Efficient defense using evidence from the shadow LoRA branch.

## Citation

Please cite our paper if you find the repository helpful.

```bibtex
@article{2026Safe-FedLLM,
  title   = {Safe-FedLLM: Delving into the Safety of Federated Large Language Models},
  author  = {Mingxiang Tao and Yu Tian and Wenxuan Tu and Yue Yang and Xue Yang and Xiangyan Tang},
  journal = {arXiv preprint arXiv:2601.07177},
  year    = {2026}
}
```

