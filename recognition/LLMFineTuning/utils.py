"""Shared utilities for training and inference.

Contains:
- decode_labels: replace -100 with pad_id for decoding
- compute_rouge: generate predictions and compute ROUGE metrics
- move_to_device: move a batch dict to device
- count_parameters: total and trainable parameter counts
- format_input_text: format raw source text for model type

Usage: imported by train.py (training loop) and predict.py (evaluation/generation).
"""

from typing import Dict, List, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import evaluate


def decode_labels(labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Replace ignore index (-100) with pad token id for decoding."""
    labels = labels.clone()
    labels[labels == -100] = pad_id
    return labels


def compute_rouge(
    model,
    tokenizer,
    dataloader: DataLoader,
    device,
    fp16: bool = False,
    num_beams: int = 4,
    max_new_tokens: int = 128,
) -> Dict[str, float]:
    """Compute ROUGE metrics for a model on a dataset.

    Args:
        model: HF model with generate
        tokenizer: matching tokenizer
        dataloader: yields dict with input_ids, attention_mask, labels
        device: torch.device
        fp16: use autocast during generation on CUDA
        num_beams: beams for generation
        max_new_tokens: max generated tokens
    Returns:
        Dict of rouge1, rouge2, rougeL, rougeLsum as floats
    """
    rouge = evaluate.load("rouge")
    model.eval()
    preds: List[str] = []
    refs: List[str] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            # Some torch type checkers don't recognize device_type kwarg; keep portable signature
            with autocast(enabled=torch.cuda.is_available() and fp16):
                gen = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                )
            pred_text = tokenizer.batch_decode(gen, skip_special_tokens=True)

            ref_ids = decode_labels(batch["labels"], tokenizer.pad_token_id)
            ref_text = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

            preds.extend(pred_text)
            refs.extend(ref_text)

    # Filter out empty references to avoid misleading zeros if dataset has empty records
    pairs = [(p, r) for p, r in zip(preds, refs) if r and r.strip()]
    if not pairs:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    pred_f, ref_f = zip(*pairs)
    scores = rouge.compute(predictions=list(pred_f), references=list(ref_f), use_stemmer=True)
    return {k: float(v) for k, v in (scores or {}).items()}


def move_to_device(batch: Dict, device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def count_parameters(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_input_text(raw: str, model_type: str, prefix: str = "summarize for layperson: ") -> str:
    """Format a raw expert report into model-ready prompt text.

    - For encoder-decoder (T5/FLAN-T5): prepend prefix
    - For decoder-only (GPT-2): use "Source: ...\n\nSummary: " pattern (no target)
    """
    if model_type == "encoder-decoder":
        return prefix + raw
    else:
        return f"Source: {raw}\n\nSummary: "
