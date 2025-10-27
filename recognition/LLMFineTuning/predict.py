"Prediction and evaluation script for fine-tuned LLMs."

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, cast

import torch
from torch.utils.data import DataLoader
import json

import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

from dataset import get_dataloaders
from modules import get_model_type
from utils import compute_rouge, decode_labels, count_parameters, move_to_device, format_input_text


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


def sample_predictions(model, tokenizer, dataloader: DataLoader, device, k: int = 5, num_beams=4, max_new_tokens=128) -> List[Dict[str, str]]:
	model.eval()
	out: List[Dict[str, str]] = []
	with torch.no_grad():
		for batch in dataloader:
			batch = move_to_device(batch, device)
			gen_ids = model.generate(
				input_ids=batch["input_ids"],
				attention_mask=batch["attention_mask"],
				num_beams=num_beams,
				max_new_tokens=max_new_tokens,
			)
			preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
			refs = tokenizer.batch_decode(decode_labels(batch["labels"], tokenizer.pad_token_id), skip_special_tokens=True)
			srcs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

			for s, p, r in zip(srcs, preds, refs):
				out.append({"source": s, "prediction": p, "reference": r})
				if len(out) >= k:
					return out
	return out

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

	if args.mode == "eval":
		# Build dataloaders (we only need one split, but get_dataloaders builds all)
		train_loader, val_loader, test_loader = get_dataloaders(
			tokenizer,
			batch_size=args.batch_size,
			max_source_length=args.max_source_length,
			max_target_length=args.max_target_length,
			model_type=model_type,
			num_workers=0,
		)
		dl_map = {"train": train_loader, "validation": val_loader, "test": test_loader}
		dataloader = dl_map[args.split]

		# Compute ROUGE
		scores = compute_rouge(
			model,
			tokenizer,
			dataloader,
			device,
			fp16=False,
			num_beams=args.num_beams,
			max_new_tokens=args.max_new_tokens,
		)

		total_params, trainable_params = count_parameters(model)

		result = {
			"model_path": args.model_path,
			"base_model_name": args.base_model_name,
			"split": args.split,
			"model_type": model_type,
			"num_beams": args.num_beams,
			"max_new_tokens": args.max_new_tokens,
			"rouge": scores,
			"parameters": {
				"total": total_params,
				"trainable": trainable_params,
			},
		}

		print(json.dumps(result, indent=2))

		# Save metrics if requested
		if args.metrics_out:
			Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
			with open(args.metrics_out, "w", encoding="utf-8") as f:
				json.dump(result, f, indent=2)

		# Qualitative samples
		samples = sample_predictions(
			model,
			tokenizer,
			dataloader,
			device,
			k=args.samples,
			num_beams=args.num_beams,
			max_new_tokens=args.max_new_tokens,
		)

		if samples:
			print("\nSample predictions:")
			for i, ex in enumerate(samples, 1):
				print(f"[{i}] Source: {ex['source'][:200].replace('\n',' ')}")
				print(f"    Pred:   {ex['prediction'][:200].replace('\n',' ')}")
				print(f"    Ref:    {ex['reference'][:200].replace('\n',' ')}")

		if args.samples_out:
			outp = Path(args.samples_out)
			outp.parent.mkdir(parents=True, exist_ok=True)
			with open(outp, "w", encoding="utf-8") as f:
				for ex in samples:
					f.write(json.dumps(ex) + "\n")
	else:
		# Generate from a single input (stdin or --input_text)
		raw = args.input_text
		if raw is None:
			try:
				raw = input().strip()
			except EOFError:
				raw = ""
		if not raw:
			print("", end="")
			return

		prompt = format_input_text(raw, model_type)
		inputs = tokenizer(
			prompt,
			return_tensors="pt",
			truncation=True,
			max_length=args.max_source_length,
		)
		inputs = {k: v.to(device) for k, v in inputs.items()}
		with torch.no_grad():
			gen = model.generate(
				**inputs,
				num_beams=args.num_beams,
				max_new_tokens=args.max_new_tokens,
			)
		out_text = tokenizer.decode(gen[0], skip_special_tokens=True)
		print(out_text)


if __name__ == "__main__":
	main()