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

loggig.basicConfig(level=logging.INFO)
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

def load_model_and_tokenizer():
    pass

def prepare_model_for_training():
    pass


def create_optimizer():
    pass


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
