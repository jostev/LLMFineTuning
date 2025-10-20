# LLM Expert Radiology Report to Layperson Summary Translator
As the title describes, this project involves fine tuning a pre-existing large language model (LLM) that will translate an expert radiology report into a layperson summary.

For specific details of use and creation:
- Dataset: BioLay Summ dataset
- Evaluation: ROUGE scores (rouge1, rouge2, rougeL, rougeLsum)
- Model: 
- Parameter count: 
- Fine-tuning strategy: 
- GPU type:
- VRAM: 
- Epochs: 
- Total training time: 

TODO: include 3-5 representative input–output examples with a short error analysis paragraph.

To run:
1. Create a virtual environment
2. Install packages in `requirements.txt`

TODO LIST:
- ROUGE eval
- predict.py
- qualitative examples
- plot and persist training curves
- log experiment metadata
- tiny ablation
- clean repo structure
- finish README.md
- polish
    - check determinism
    - decode sanity
    - save to single checkpoint folder
- ensure proper commits
