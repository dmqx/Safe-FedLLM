import warnings
import logging

def suppress_training_warnings():

    warnings.filterwarnings("ignore", message=".*will be cast from.*to float16 during quantization.*")
    warnings.filterwarnings("ignore", message=".*MatMul8bitLt: inputs will be cast.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
    warnings.filterwarnings("ignore", message=".*use_cache=True.*is incompatible with gradient checkpointing.*")
    warnings.filterwarnings("ignore", message=".*This instance will be ignored in loss calculation.*")
    warnings.filterwarnings("ignore", message=".*consider increasing the.*max_seq_length.*")
    warnings.filterwarnings("ignore", message=".*Could not find response key.*in the following instance.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="trl")
    warnings.filterwarnings("ignore", category=UserWarning, module="trl.trainer.utils")
    warnings.filterwarnings("ignore",category=FutureWarning,message=r".*`tokenizer` is deprecated and will be removed in version 5\.0\.0 for `SFTTrainer.__init__`.*")
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour.*")
    warnings.filterwarnings("ignore", message=".*The attention mask and the pad token id were not set.*")
    warnings.filterwarnings("ignore", message=".*This implementation of AdamW is deprecated.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")
    warnings.filterwarnings("ignore", message=".*target_modules.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="peft")
    warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    logging.getLogger("trl").setLevel(logging.WARNING)

def enable_all_warnings():
    warnings.resetwarnings()

if __name__ == "__main__":
    suppress_training_warnings()
    print("Warning suppressor is enabled")
