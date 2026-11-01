# Instruction-Tuning-LLM

Dialogue summarization with FLAN-T5 on DialogSum. One notebook compares prompting, full fine-tuning, LoRA, and a short PPO run that rewards less toxic summaries.

## What's inside

- `notebooks/InstructionFineTuning.ipynb` is the main notebook
- `utils/` holds prompt, data and metric helpers

## How to run

```bash
pip install -r requirements.txt "trl==0.11.4" "transformers==4.45.2"
jupyter notebook notebooks/InstructionFineTuning.ipynb
```

## Related tutorial

- [Adapters and LoRA](https://raminaeye.github.io/adapter.html)
