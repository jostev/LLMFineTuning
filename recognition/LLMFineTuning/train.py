"""Training script for LLM fine-tuning."""
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from modules import *
from dataset import get_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--output_dir", type=str, default="runs/project13")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--scheduler_type", type=str, default="linear", choices=["linear","cosine"])
    parser.add_argument("--optimizer_type", type=str, default="adamw", choices=["adamw","adam","sgd"])
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()