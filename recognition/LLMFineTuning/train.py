"""Training script for LLM fine-tuning."""
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import evaluate
from modules import *
from dataset import get_dataloaders
from tqdm.auto import tqdm

rouge = evaluate.load("rouge")

def decode_labels(labels, pad_id):
    """Decode labels by replacing -100 with pad_id."""
    labels = labels.clone()
    labels[labels == -100] = pad_id
    return labels

def compute_rouge(model, tokenizer, dataloader, device, fp16, num_beams=4, max_new_tokens=128):
    """Compute ROUGE scores on the dataset using the model."""
    model.eval()  # Set model to evaluation mode
    preds = []
    refs = []
    with torch.no_grad():
        for batch in dataloader:
            # Move any tensor values in batch to target device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Generate predictions from model
            with autocast(enabled=torch.cuda.is_available() and fp16):
                gen = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens
                )

            # Decode generated token ids to text
            pred_text = tokenizer.batch_decode(gen, skip_special_tokens=True)

            # Replace -100 in labels with pad token id and decode references to text
            ref_ids = decode_labels(batch["labels"], tokenizer.pad_token_id)
            ref_text = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

            preds.extend(pred_text)
            refs.extend(ref_text)

    # Compute ROUGE and convert numpy/torch types to python floats
    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {k: float(v) for k, v in scores.items()}

def move_to_device(batch, device):
    """Move batch of data to target device."""
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
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

    parser.add_argument("--eval_num_beams", type=int, default=4)
    parser.add_argument("--eval_max_new_tokens", type=int, default=128)
    parser.add_argument("--select_by", type=str, default="loss", choices=["loss","rougeL"])

    args = parser.parse_args()

    # Validate 8-bit training requirement
    if args.use_8bit and not torch.cuda.is_available():
        print("8-bit training requires a CUDA-capable device. Disabling 8-bit mode.")
        args.use_8bit = False

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
    scaler = GradScaler(enabled=args.fp16)

    # Prepare output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    # Prepare variables
    best_val = float("inf")
    best_rougel = -1.0

    # Training loop
    print("Starting training...")
    if torch.cuda.is_available():
        print(torch.cuda.mem_get_info())
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = move_to_device(batch, device)

            # Forward pass with mixed precision    
            with autocast(enabled=args.fp16 and torch.cuda.is_available()):
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

        # End of epoch evaluation: compute loss and ROUGE          
        val_loss = evaluate(model, val_loader, device)
        metrics = compute_rouge(
            model, tokenizer, val_loader, device, args.fp16,
            num_beams=args.eval_num_beams,
            max_new_tokens=args.eval_max_new_tokens
        )

        # Epoch logging
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {running/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save per-epoch model
        ckpt_dir = out_dir / f"epoch{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir.as_posix())
        tokenizer.save_pretrained(ckpt_dir.as_posix())

        row = {
            "epoch": epoch,
            "train_avg_loss": round(running / max(1, len(train_loader)), 6),
            "val_loss": round(val_loss, 6),
            **{k: round(v, 6) for k, v in metrics.items()}
        }
        with open((out_dir / "metrics.jsonl").as_posix(), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        with open((ckpt_dir / "metrics.json").as_posix(), "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        
        # Track best model
        key = "val_loss" if args.select_by == "loss" else "rougeL"
        better = ((val_loss < best_val) 
                  if args.select_by == "loss" 
                  else (metrics.get("rougeL", 0.0) > locals().get("best_rougel", 0.0)))
        if better:
            if args.select_by == "loss":
                best_val = val_loss
            else:
                best_rougel = metrics.get("rougeL", 0.0)
            for f in best_dir.iterdir():
                if f.is_file():
                    f.unlink()
            model.save_pretrained(best_dir.as_posix())
            tokenizer.save_pretrained(best_dir.as_posix())
            with open((best_dir / "metrics.json").as_posix(), "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
