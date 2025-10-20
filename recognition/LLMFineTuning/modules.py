"""
Model initialization and configuration utilities for fine-tuning LLMs.
Supports encoder-decoder models (T5, FLAN-T5) and decoder-only models (GPT-2).
"""

import torch
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from typing import Dict, Optional, Tuple
from peft import get_peft_model, LoraConfig, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model registry with supported models
ENCODER_DECODER_MODELS = {
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
}

DECODER_ONLY_MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
}

ALL_MODELS = {**ENCODER_DECODER_MODELS, **DECODER_ONLY_MODELS}


def get_model_type(model_name: str) -> str:
    """
    Determine if a model is encoder-decoder or decoder-only.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        "encoder-decoder" or "decoder-only"
    """
    if model_name in ENCODER_DECODER_MODELS or model_name in ENCODER_DECODER_MODELS.values():
        return "encoder-decoder"
    elif model_name in DECODER_ONLY_MODELS or model_name in DECODER_ONLY_MODELS.values():
        return "decoder-only"
    else:
        # Try to infer from model name
        if any(name in model_name.lower() for name in ["t5", "bart", "pegasus"]):
            return "encoder-decoder"
        elif any(name in model_name.lower() for name in ["gpt", "gpt2", "gpt-neo"]):
            return "decoder-only"
        else:
            raise ValueError(
                f"Unknown model type for {model_name}. "
                f"Supported models: {list(ALL_MODELS.keys())}"
            )


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_8bit: bool = false,
    gradient_checkpointing: bool = false
) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    """
    Load a pre-trained model and tokenizer.
    
    Args:
        model_name: Model name or path (e.g., "t5-small", "gpt2", "flan-t5-base")
        device: Device to load model on ("cuda" or "cpu")
        use_8bit: Whether to use 8-bit quantization (requires bitsandbytes)
        gradient_checkpointing: Enable gradient checkpointing to save memory
        
    Returns:
        Tuple of (model, tokenizer, model_type)
    """
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")
    
    # Get full model path if using shorthand
    model_path = ALL_MODELS.get(model_name, model_name)
    model_type = get_model_type(model_name)
    
    logger.info(f"Model type: {model_type}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configure model loading arguments
    model_kwargs = {}
    if use_8bit:
        try:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
            logger.info("Using 8-bit quantization")
        except Exception as e:
            logger.warning(f"8-bit quantization not available: {e}")
            use_8bit = False
    
    # Load model based on type
    if model_type == "encoder-decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
    else:  # decoder-only
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Set pad token for decoder-only models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Enable gradient checkpointing if requested
    if gradient_checkpointing and not use_8bit:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Move model to device if not using 8-bit (8-bit handles device placement)
    if not use_8bit:
        model = model.to(device)
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, tokenizer, model_type


def prepare_model_for_training(
    model: torch.nn.Module,
    freeze_encoder: bool = False,
    freeze_embeddings: bool = False,
    lora_config: Optional[Dict] = None
) -> torch.nn.Module:
    """
    Prepare model for fine-tuning with optional parameter freezing or LoRA.
    
    Args:
        model: Pre-trained model
        freeze_encoder: Freeze encoder layers (for encoder-decoder models)
        freeze_embeddings: Freeze embedding layers
        lora_config: Configuration for LoRA (Low-Rank Adaptation)
                    Example: {"r": 8, "lora_alpha": 32, "lora_dropout": 0.1}
        
    Returns:
        Prepared model
    """


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    optimizer_type: str = "adamw"
) -> torch.optim.Optimizer:
    """
    Create optimizer for training.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        adam_epsilon: Epsilon for Adam optimizer
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
        
    Returns:
        Optimizer instance
    """
    # Prepare optimizer grouped parameters
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon
        )
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unimplemented optimizer type: {optimizer_type}")
    
    logger.info(f"Optimizer created: {optimizer_type.upper()}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "linear"
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps (if None, uses warmup_ratio)
        warmup_ratio: Ratio of warmup steps to total steps
        scheduler_type: Type of scheduler ("linear" or "cosine")
        
    Returns:
        Learning rate scheduler
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    if scheduler_type.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type.lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unimplemented scheduler type: {scheduler_type}")

    logger.info(f"Scheduler created: {scheduler_type}")
    logger.info(f"Warmup steps: {num_warmup_steps} / {num_training_steps}")
    
    return scheduler


class ModelConfig:
    """Configuration class for model training."""
    
    def __init__(
        self,
        model_name: str = "t5-small",
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        num_epochs: int = 3,
        max_source_length: int = 512,
        max_target_length: int = 256,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        save_total_limit: int = 3,
        output_dir: str = "./output",
        use_8bit: bool = False,
        gradient_checkpointing: bool = False,
        freeze_encoder: bool = False,
        freeze_embeddings: bool = False,
        lora_config: Optional[Dict] = None,
        optimizer_type: str = "adamw",
        scheduler_type: str = "linear",
        fp16: bool = False,
        num_beams: int = 4,
        early_stopping: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize model configuration.
        
        Args:
            model_name: Name of the pre-trained model
            learning_rate: Learning rate for optimization
            batch_size: Training batch size
            num_epochs: Number of training epochs
            max_source_length: Maximum length for source sequences
            max_target_length: Maximum length for target sequences
            weight_decay: Weight decay for regularization
            warmup_ratio: Ratio of warmup steps
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            seed: Random seed for reproducibility
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps
            save_total_limit: Maximum number of checkpoints to keep
            output_dir: Directory to save outputs
            use_8bit: Use 8-bit quantization
            gradient_checkpointing: Enable gradient checkpointing
            freeze_encoder: Freeze encoder layers
            freeze_embeddings: Freeze embedding layers
            lora_config: LoRA configuration dictionary
            optimizer_type: Type of optimizer
            scheduler_type: Type of learning rate scheduler
            fp16: Use mixed precision training
            num_beams: Number of beams for generation
            early_stopping: Use early stopping in generation
            device: Device to use (None for auto-detection)
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        self.output_dir = output_dir
        self.use_8bit = use_8bit
        self.gradient_checkpointing = gradient_checkpointing
        self.freeze_encoder = freeze_encoder
        self.freeze_embeddings = freeze_embeddings
        self.lora_config = lora_config
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.fp16 = fp16 and torch.cuda.is_available()
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = get_model_type(model_name)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        config_str = "ModelConfig(\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}={value},\n"
        config_str += ")"
        return config_str
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed}")
