"""Prediction and evaluation script for fine-tuned LLMs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, cast

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

from modules import get_model_type


def load_model_any(model_path: Path, base_model_name: Optional[str] = None):
	"""Load either a full model directory or a LoRA adapter on top of a base model.

	Heuristic: if adapter_config.json exists in model_path, treat as LoRA adapter.
	"""
	adapter_cfg = model_path / "adapter_config.json"

	if adapter_cfg.exists():
		# LoRA adapter case
		if base_model_name is None:
			raise ValueError("Detected LoRA adapter directory but --base_model_name was not provided.")
		model_type = get_model_type(base_model_name)
		if model_type == "encoder-decoder":
			base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
		else:
			base = AutoModelForCausalLM.from_pretrained(base_model_name)
		try:
			from peft import PeftModel  # type: ignore
		except ImportError:
			raise RuntimeError("peft is required to load LoRA adapters. Install with `pip install peft`. ")
		model = PeftModel.from_pretrained(base, model_path.as_posix())
		tokenizer = AutoTokenizer.from_pretrained(base_model_name)
		return model, tokenizer, model_type

	# Full model directory or HF hub id
	model_id = model_path.as_posix()
	# Infer model type from name
	try:
		model_type = get_model_type(model_id)
	except Exception:
		# Fallback to base_model_name if provided
		if base_model_name:
			model_type = get_model_type(base_model_name)
		else:
			# Default guess: encoder-decoder for T5-like
			model_type = "encoder-decoder"

	if model_type == "encoder-decoder":
		model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
	else:
		model = AutoModelForCausalLM.from_pretrained(model_id)

	tokenizer = AutoTokenizer.from_pretrained(model_id)
	if tokenizer.pad_token is None and hasattr(model.config, "eos_token_id"):
		tokenizer.pad_token = tokenizer.eos_token
	return model, tokenizer, model_type


def main():
	parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model and run predictions.")
	parser.add_argument("--model_path", type=str, required=True, help="Path to model dir (or HF id). If LoRA adapter, pass directory containing adapter_config.json")
	parser.add_argument("--base_model_name", type=str, default=None, help="Base model name if loading a LoRA adapter (e.g., google/flan-t5-base)")
	parser.add_argument("--mode", type=str, default="eval", choices=["eval", "generate"], help="Run dataset evaluation or single-text generation")
	parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split for eval mode")
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--max_source_length", type=int, default=512)
	parser.add_argument("--max_target_length", type=int, default=256)
	parser.add_argument("--num_beams", type=int, default=4)
	parser.add_argument("--max_new_tokens", type=int, default=128)
	parser.add_argument("--input_text", type=str, default=None, help="Raw expert radiology report for generate mode; falls back to stdin if not provided")
	parser.add_argument("--samples", type=int, default=5, help="Number of qualitative samples to save/print")
	parser.add_argument("--samples_out", type=str, default=None, help="Optional path to save samples JSONL")
	parser.add_argument("--metrics_out", type=str, default=None, help="Optional path to save metrics JSON")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_path = Path(args.model_path)
	model, tokenizer, model_type = load_model_any(model_path, base_model_name=args.base_model_name)
	cast(nn.Module, model).to(device)

	print(f"Successfully loaded model {args.model_path} ({model_type}) on {device}.")


if __name__ == "__main__":
	main()
