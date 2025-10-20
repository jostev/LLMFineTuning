"""Training script for LLM fine-tuning."""
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from modules import *
from dataset import get_dataloaders
from tqdm.auto import tqdm

def move_to_device(batch, device):
    """Move batch of data to target device."""
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True)
    return out

def evaluate(model, dataloader, device):
    """Evaluate model on validation dataset."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            batch_size = batch['input_ids'].size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
    return total_loss / max(1, total_count)
    

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

    model.config.use_cache = False  # Disable cache for training

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
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=args.fp16)

    # Prepare output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("Starting training...")
    print(torch.cuda.mem_get_info())
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = move_to_device(batch, device)

            # Forward pass with mixed precision    
            with torch.amp.autocast(device_type="cuda", enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            # Backward pass with gradient scaling   
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum

            # Optimiser step after gradient accumulation
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{running / max(1, step):.4f}"})

        print(torch.cuda.mem_get_info())

        # End of epoch evaluation            
        val_loss = evaluate(model, val_loader, device, fp16=args.fp16)

        # Epoch logging
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {running/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save per-epoch model
        ckpt_dir = out_dir / f"epoch{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir.as_posix())
        tokenizer.save_pretrained(ckpt_dir.as_posix())

        # Append log
        with open((out_dir / "log.txt").as_posix(), "a", encoding="utf-8") as f:
            f.write(
                f"epoch\t{epoch}\ttrain_avg_loss\t{running / max(1, len(train_loader)):.6f}\tval_loss\t{val_loss:.6f}\n"
            )
        
        # Track best model
        if val_loss < best_val:
            best_val = val_loss
            for f in best_dir.iterdir():
                if f.is_file():
                    f.unlink()
            model.save_pretrained(best_dir.as_posix())
            tokenizer.save_pretrained(best_dir.as_posix())


if __name__ == "__main__":
    main()