import os
import torch
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class ModelConfig:
    """Configuration for model loading and LoRA setup"""
    model_name: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    dtype: torch.dtype = torch.bfloat16

@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation"""
    r: int = 32
    target_modules: List[str] = None
    lora_alpha: int = 32
    lora_dropout: float = 0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[dict] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    dataset_text_field: str = "text"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 30
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "none"

@dataclass
class DataConfig:
    """Configuration for data processing"""
    data_path: str = "data/unified_error_dataset_annotated.csv"
    chat_percentage: float = None
    shuffle_seed: int = 3407

@dataclass
class InferenceConfig:
    """Configuration for inference settings"""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    enable_thinking: bool = False
    run_sample_comparison: bool = True
    sample_random_state: int = 42

@dataclass
class SaveConfig:
    """Configuration for model saving"""
    base_output_dir: str = "output"
    use_timestamp: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    save_merged: bool = False
    save_method: str = "lora"  # Options: "merged_16bit", "merged_4bit", "lora"
    push_to_hub: bool = False
    hub_model_name: Optional[str] = None
    save_gguf: bool = False
    gguf_quantization_method: str = "q8_0"
    save_training_config: bool = True
    save_training_summary: bool = True
    
    def get_output_dir(self, model_name: str) -> str:
        """Generate output directory path with timestamp"""
        if self.use_timestamp:
            timestamp = datetime.now().strftime(self.timestamp_format)
            # Clean model name for directory
            clean_model_name = model_name.replace("/", "_").replace("-", "_")
            return os.path.join(self.base_output_dir, timestamp, f"lora_{clean_model_name}")
        else:
            clean_model_name = model_name.replace("/", "_").replace("-", "_")
            return os.path.join(self.base_output_dir, f"lora_{clean_model_name}") 