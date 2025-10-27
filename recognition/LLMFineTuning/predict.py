"""Prediction and evaluation script for fine-tuned LLMs."""

import argparse
import json
from pathlib import Path

import torch


def main():
	parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model and run predictions.")
	parser.add_argument("--model_path", type=str, required=True, help="Path to model dir (or HF id).")
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
	print(f"Using device: {device}")


if __name__ == "__main__":
	main()
