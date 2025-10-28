# Overview
This project implements a fine tuning model for pre-existing large language models (LLMs); with the goal of translating an expert radiology reports into a layperson summaries.

TODO LIST:
- finish README.md
- polish
    - check determinism
    - decode sanity
    - save to single checkpoint folder
- ensure proper commits

# Models Used (Architecture)
## Encoder-Decoder
## Decoder
## Fine-tuning

# Dataset

# Training
## Parameters
For specific details of use and creation:
- Dataset: BioLay Summ dataset
- Evaluation: ROUGE scores (rouge1, rouge2, rougeL, rougeLsum)
- Model: flan-t5 base
- Parameter count: 
- Fine-tuning strategy: 
- GPU type:
- VRAM: 
- Epochs: 
- Total training time: 
## Environment

# Results
TODO: include 3-5 representative input–output examples with a short error analysis paragraph.

## Performance Metrics
## Predictions

# Usage
## Initialisation
The example in this project was ran using python 3.14; to start with:
1. Create a virtual environment `python -m venv env-name`
2. Activate the environemnt `source env-name/bin/activate` (for UNIX), `.\env-name\Scripts\activate`
3. Install required libraries `pip install -r requirements.txt`

## Script Usage
Arguments for `train.py`:
- `--model_name` - name of the LLM to fine-tune (you can view the list in `modules.py`: t5, flan-t5 and gpt2)
- `--output_dir` - 
- `` - 
- `` - 

# References
