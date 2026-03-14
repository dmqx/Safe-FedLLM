"""
警告抑制器模块
用于统一管理和忽略联邦学习训练过程中的常见警告
"""

import warnings
import logging

def suppress_training_warnings():
    """
    抑制联邦学习训练过程中的常见警告
    """
    # 忽略 BitsAndBytes 量化警告
    warnings.filterwarnings("ignore", message=".*will be cast from.*to float16 during quantization.*")
    warnings.filterwarnings("ignore", message=".*MatMul8bitLt: inputs will be cast.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
    
    # 忽略缓存兼容性警告
    warnings.filterwarnings("ignore", message=".*use_cache=True.*is incompatible with gradient checkpointing.*")
    
    # 忽略序列长度截断警告
    warnings.filterwarnings("ignore", message=".*This instance will be ignored in loss calculation.*")
    warnings.filterwarnings("ignore", message=".*consider increasing the.*max_seq_length.*")
    
    # 忽略 TRL 相关警告
    warnings.filterwarnings("ignore", message=".*Could not find response key.*in the following instance.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="trl")
    warnings.filterwarnings("ignore", category=UserWarning, module="trl.trainer.utils")
    warnings.filterwarnings("ignore",category=FutureWarning,message=r".*`tokenizer` is deprecated and will be removed in version 5\.0\.0 for `SFTTrainer.__init__`.*")
    
    # 忽略 Transformers 相关警告
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour.*")
    warnings.filterwarnings("ignore", message=".*The attention mask and the pad token id were not set.*")
    warnings.filterwarnings("ignore", message=".*This implementation of AdamW is deprecated.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")
    
    # 忽略 PEFT 相关警告
    warnings.filterwarnings("ignore", message=".*target_modules.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="peft")
    
    # 忽略 PyTorch 相关警告
    warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def setup_logging(log_level=logging.INFO):
    """
    设置日志级别，只显示重要信息
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置特定模块的日志级别
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    logging.getLogger("trl").setLevel(logging.WARNING)

def enable_all_warnings():
    """
    重新启用所有警告（用于调试）
    """
    warnings.resetwarnings()

if __name__ == "__main__":
    # 测试警告抑制器
    suppress_training_warnings()
    print("警告抑制器已启用")
