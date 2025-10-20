"""
Dataset loader for BioLaySumm (Biomedical Lay Summarization) dataset.
This module provides utilities to load and preprocess radiology reports
for fine-tuning LLMs to translate expert reports into layperson summaries.
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset

path = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"

train_ds = load_dataset(path, split="train")
val_ds = load_dataset(path, split="validation")
test_ds = load_dataset(path, split="test")


