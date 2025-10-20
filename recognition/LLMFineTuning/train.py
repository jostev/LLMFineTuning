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
    # Set --lora_r > 0 to enable LoRA
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer, model_type = load_model_and_tokenizer(
        args.model_name,
        device=device,
        use_8bit=args.use_8bit,
        gradient_checkpointing=args.grad_checkpointing
    )

    # Configure optional LoRA and freezing
    lora_config = None
    if args.lora_r and args.lora_r > 0:
        lora_config = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout
        }
    
    # Prepare model for training
    model = prepare_model_for_training(
        model,
        freeze_encoder=args.freeze_encoder,
        freeze_embeddings=args.freeze_embeddings,
        lora_config=lora_config
    )

    # Build dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        model_type=model_type
    )

    # Compute total training steps
    total_steps = (len(train_loader) * args.epochs) // max(1, args.grad_accum)

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer_type
    )

    scheduler = create_scheduler(
        optimizer,
        num_training_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        scheduler_type=args.scheduler_type,
    )
    
    



if __name__ == "__main__":
    main()