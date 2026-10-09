# Instruction Tuning an LLM for Dialogue Summarization

Fine-tunes `google/flan-t5-base` on the DialogSum dataset for conversation summarization, comparing prompting, full instruction fine-tuning, LoRA, and a short RLHF loop with PPO.

## What this repo covers

1. **Prompting baselines**: zero-shot, one-shot, and few-shot summarization with no weight updates.
2. **Full instruction fine-tuning**: supervised training on tokenized dialogue and summary pairs.
3. **PEFT with LoRA**: the same task with low-rank adapters, a fraction of the trainable parameters, then merged back into the base model for inference.
4. **RLHF with PPO**: a toxicity classifier acts as the reward model, and PPO steers the LoRA-tuned policy toward less toxic summaries while a KL penalty keeps it close to the reference model.
5. **Evaluation**: ROUGE scores compare generated summaries against human-written references for each approach.

## Repo structure

```
.
├── notebooks/
│   └── InstructionFineTuning.ipynb   # Main notebook: prompting, fine-tuning, LoRA, RLHF, evaluation
├── utils/
│   ├── helper.py                     # Prompt building, tokenization, PPO dataset prep, toxicity eval
│   ├── data_utility.py               # Lightweight text cleaning helpers
│   └── metrics.py                    # Reference ROUGE / BLEU / LCS implementations with unit tests
├── snapshots/
│   └── ppo.png                       # Diagram of the PPO setup used in the notebook
├── requirements.txt                  # Python dependencies
└── README.md
```

## How to run it

Requires Python 3.8 or newer and a GPU for training. The notebook downloads models and datasets from the HuggingFace Hub, so you need a network connection and a HuggingFace token for gated models.

```bash
git clone https://github.com/raminaeye/Instruction-Tuning-LLM.git
cd Instruction-Tuning-LLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/InstructionFineTuning.ipynb
```

Run the cells top to bottom. Training steps are intentionally short (a few steps each) so the notebook works as a demo; increase `max_steps`, `num_train_epochs`, and `max_ppo_steps` for real runs. The `utils/metrics.py` unit tests run standalone with `python utils/metrics.py` and need no extra dependencies.

## What you learn

How to prompt a seq2seq model for summarization, how to instruction fine-tune it fully and with LoRA, how PPO with a classifier-based reward model reduces toxicity, and how ROUGE quantifies each step. The `utils/helper.py` module shows the reusable pieces: prompt construction, batched tokenization for `datasets.map`, PPO dataset preparation, and a small generation and toxicity scoring loop.
