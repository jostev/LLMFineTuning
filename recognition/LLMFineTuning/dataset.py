"""
Dataset loader for BioLaySumm (Biomedical Lay Summarization) dataset.
This module provides utilities to load and preprocess radiology reports
for fine-tuning LLMs to translate expert reports into layperson summaries.
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import Dict, List, Optional

# BioLaySumm dataset path for ACL 2025 BioLaySumm workshop Subtask 2.1
DATASET_PATH = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"


def load_biolaysumm_data(split: str) -> Dataset:
    """
    Load BioLaySumm dataset split.
    
    Args:
        split: Dataset split to load ('train', 'validation', or 'test')
    
    Returns:
        Hugging Face dataset object
    """
    return load_dataset(DATASET_PATH, split=split)


class BioLaySummDataset(Dataset):
    """
    PyTorch Dataset for BioLaySumm radiology report summarization.
    Supports both encoder-decoder models (T5, FLAN-T5) and decoder-only models (GPT-2).
    """
    
    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 256,
        model_type: str = "encoder-decoder",
        prefix: str = "summarize for layperson: "
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset: Hugging Face dataset object
            tokenizer: Tokenizer for the model
            max_source_length: Maximum length for input sequences
            max_target_length: Maximum length for target sequences
            model_type: Either "encoder-decoder" (T5/FLAN-T5) or "decoder-only" (GPT-2)
            prefix: Task prefix to prepend to source text (for T5/FLAN-T5)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type.lower()
        self.prefix = prefix
        
        # Ensure decoder-only models have a pad token
        if (self.model_type == "decoder-only" 
            and self.tokenizer.pad_token is None):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single preprocessed example.
        
        Returns:
            Dictionary with tokenized inputs and labels
        """
        example = self.dataset[idx]
        
        # Extract source and target
        # + adjust field names based on BioLaySumm dataset structure
        source_text = example.get("radiology_report", "")
        target_text = example.get("layman_report", "")
        
        if self.model_type == "encoder-decoder":
            return self._encode_encoder_decoder(source_text, target_text)
        elif self.model_type == "decoder-only":
            return self._encode_decoder_only(source_text, target_text)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _encode_encoder_decoder(self, source: str, target: str) -> Dict[str, torch.Tensor]:
        """
        Encode for encoder-decoder models (T5, FLAN-T5).
        
        Args:
            source: Expert radiology report
            target: Layperson summary
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        source_with_prefix = self.prefix + source
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source_with_prefix,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Replace padding token ids with -100 for loss calculation
        labels = target_encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
    
    def _encode_decoder_only(self, source: str, target: str) -> Dict[str, torch.Tensor]:
        """
        Encode for decoder-only models (GPT-2).
        
        For GPT-2, we concatenate source and target with special separators.
        Format: "Source: <source text> Summary: <target text><eos>"
        
        Args:
            source: Expert radiology report
            target: Layperson summary
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Format: "Source: {source} Summary: {target}"
        full_text = f"Source: {source}\n\nSummary: {target}{self.tokenizer.eos_token}"
        
        # Tokenize the full sequence
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_source_length + self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For causal language modeling, input_ids and labels are the same
        # but we need to mask the source part in labels
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Create labels: -100 for source tokens, actual token ids for summary
        labels = input_ids.clone()
        
        # Find where "Summary:" starts to only compute loss on summary tokens
        source_prompt = f"Source: {source}\n\nSummary: "
        source_tokens = self.tokenizer(
            source_prompt,
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze()
        
        # Mask source tokens in labels
        source_length = len(source_tokens)
        labels[:source_length] = -100
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def get_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_source_length: int = 512,
    max_target_length: int = 256,
    model_type: str = "encoder-decoder",
    num_workers: int = 0
):
    """
    Create DataLoaders for train, validation, and test splits.
    
    Args:
        tokenizer: Tokenizer for the model
        batch_size: Batch size for DataLoader
        max_source_length: Maximum length for input sequences
        max_target_length: Maximum length for target sequences
        model_type: Either "encoder-decoder" or "decoder-only"
        num_workers: Number of workers for DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Load datasets
    train_ds = load_biolaysumm_data("train")
    val_ds = load_biolaysumm_data("validation")
    test_ds = load_biolaysumm_data("test")
    
    # Create Datasets
    train_dataset = BioLaySummDataset(
        train_ds,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        model_type=model_type
    )
    
    val_dataset = BioLaySummDataset(
        val_ds,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        model_type=model_type
    )
    
    test_dataset = BioLaySummDataset(
        test_ds,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        model_type=model_type
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
