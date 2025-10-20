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
            raise ValueError("Unknown model.")

def load_model_and_tokenizer(model_name, device, use_8bit=False, gradient_checkpointing=False):
    # Initialise device
    device = device if torch.cuda.is_available() else "cpu"

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
            logger.warning(f"Error loading 8-bit quantization: {e}")
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

def prepare_model_for_training(model, freeze_encoder, freeze_embeddings, lora_config):
    if lora_config:
        # Determine task type
        if hasattr(model, "encoder"):
            task_type = TaskType.SEQ_2_SEQ_LM
        else:
            task_type = TaskType.CAUSAL_LM
        
        peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            target_modules=lora_config.get("target_modules", ["q", "v"])
        )
        
        model = get_peft_model(model, peft_config)
        logger.info("LoRA enabled")
        model.print_trainable_parameters()
        return model
    
    # Freeze encoder if requested (for encoder-decoder models)
    if freeze_encoder and hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    # Freeze embeddings if requested
    if freeze_embeddings:
        if hasattr(model, "shared"):  # T5 models
            for param in model.shared.parameters():
                param.requires_grad = False
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):  # GPT-2
            for param in model.transformer.wte.parameters():
                param.requires_grad = False
            for param in model.transformer.wpe.parameters():
                param.requires_grad = False
        logger.info("Embeddings frozen")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    return model


def create_optimizer(model, learning_rate=5e-5, weight_decay=0.01, adam_epsilon=1e-8, optimizer_type="adamw"):
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


def create_scheduler():
    pass

class ModelConfig:
    """Configuration class for model training."""
    
    def __init__():
        """
        Initialize model configuration.
        """

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
