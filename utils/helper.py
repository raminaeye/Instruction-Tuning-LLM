"""Helpers for the instruction fine-tuning notebook.

Covers prompt construction, dialogue summarization data tokenization,
LoRA/PPO dataset preparation, and a small toxicity evaluation loop.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
import torch
import numpy as np


class Helper():
    """Shared helpers for the InstructionFineTuning notebook.

    The DialogSum dataset is loaded lazily (on first use, not at import
    time) so that importing this module never triggers a download.
    """

    def __init__(self, tokenizer, dataset_name='knkarthick/dialogsum', split='test'):
        """Store the tokenizer and dataset location.

        Args:
            tokenizer: A HuggingFace tokenizer used for prompts and outputs.
            dataset_name: HuggingFace Hub id of the dialogue dataset.
            split: Dataset split used for prompt construction and evaluation.
        """
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self._dataset = None

    @property
    def dataset(self):
        """The loaded DialogSum dataset, downloaded on first access."""
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_name)
        return self._dataset

    def create_prompt(self, example_indicies, target_index=1):
        """Build a few-shot summarization prompt for one dialogue.

        The prompt starts with the given example dialogues paired with their
        summaries, then appends the target dialogue followed by "Summary:"
        so the model completes the summary.

        Args:
            example_indicies: Indices of example dialogues shown as context.
            target_index: Index of the dialogue to summarize.

        Returns:
            A tuple (prompt, ground_truth_summary).
        """
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary:'

        prompt = ''
        for index in example_indicies:
            dialogue = self.dataset[self.split][index]['dialogue']
            summary = self.dataset[self.split][index]['summary']
            prompt += start_prompt + dialogue + end_prompt + summary + '\n'

        dialogue = self.dataset[self.split][target_index]['dialogue']
        ground_truth = self.dataset[self.split][target_index]['summary']
        prompt += start_prompt + dialogue + end_prompt

        return prompt, ground_truth

    def gen_output(self, prompt, original_model, fine_tuned_model):
        """Generate summaries with the base and the fine-tuned model.

        Args:
            prompt: The summarization prompt.
            original_model: The untouched base model.
            fine_tuned_model: The fine-tuned model to compare against.

        Returns:
            A tuple (original_output, fine_tuned_output) with the prompt
            echoed text stripped from each generation.
        """
        inputs = self.tokenizer(prompt, return_tensors='pt')
        output_o = self.tokenizer.decode(
            original_model.generate(
                inputs['input_ids'], do_sample=True, temperature=0.5)[0],
            skip_special_tokens=True)
        output_f = self.tokenizer.decode(
            fine_tuned_model.generate(
                input_ids=inputs['input_ids'], do_sample=True,
                temperature=0.5)[0],
            skip_special_tokens=True)

        return (' '.join(output_o.split('Dialogue:')[-1:]),
                ' '.join(output_f.split('Dialogue:')[-1:]))

    def tokenizer_function(self, example):
        """Tokenize a batch of dialogues and summaries for seq2seq training.

        Works with datasets.map(batched=True): inputs and labels are returned
        as lists of token id lists (no return_tensors, which would produce a
        single stacked tensor and break batched mapping).

        Args:
            example: A batch dict with 'dialogue' and 'summary' lists.

        Returns:
            The batch with 'input_ids' and 'labels' added.
        """
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary:'
        prompt = [start_prompt + dialogue + end_prompt
                  for dialogue in example['dialogue']]
        example['input_ids'] = self.tokenizer(
            prompt, padding='max_length', truncation=True).input_ids
        example['labels'] = self.tokenizer(
            example['summary'], padding='max_length', truncation=True).input_ids

        return example

    def build_dataset(self, model_name, dataset_name, minLen, maxLen):
        """Build the PPO training dataset with prompt and query fields.

        Filters dialogues by character length, tokenizes each prompt, keeps a
        decoded 'query' string for reward computation, and returns a
        train/test split formatted for torch.

        Args:
            model_name: Model whose tokenizer is used.
            dataset_name: HuggingFace Hub id of the dataset.
            minLen: Minimum dialogue length (characters).
            maxLen: Maximum dialogue length (characters).

        Returns:
            A DatasetDict with 'train' and 'test' splits.
        """
        dataset = load_dataset(dataset_name, split='train')
        dataset = dataset.filter(
            lambda x: len(x['dialogue']) > minLen
            and len(x['dialogue']) <= maxLen)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = self._inference_device()

        def tokenize(sample):
            """Tokenize one dialogue into input_ids and a decoded query."""
            prompt = ('Summarize the following conversation.\n'
                      f"{sample['dialogue']}\n\nSummary:\n")
            sample['input_ids'] = tokenizer(
                prompt, return_tensors='pt').input_ids.to(device)
            sample['query'] = tokenizer.decode(sample['input_ids'][0])
            return sample

        dataset = dataset.map(tokenize, batched=False)
        dataset.set_format(type='torch')
        dataset_splits = dataset.train_test_split(
            test_size=0.2, shuffle=False, seed=42)
        return dataset_splits

    @staticmethod
    def _inference_device():
        """Pick the best available torch device (mps, cuda, or cpu)."""
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def evaluate_toxic(self, model, toxic_eval, tokenizer, dataset, num_samples):
        """Measure mean and std of toxicity scores over dataset samples.

        Generates a continuation for each input and scores the combined
        input plus generation with the toxicity measurement module.

        Args:
            model: The model to evaluate.
            toxic_eval: An evaluate toxicity measurement module.
            tokenizer: Tokenizer matching the model.
            dataset: Dataset of dicts with a 'query' field.
            num_samples: How many samples to score.

        Returns:
            A tuple (mean_toxicity, std_toxicity).
        """
        max_new_tokens = 100
        toxics = []
        input_texts = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            input_text = sample["query"]
            input_texts.append(input_text)
            input_ids = tokenizer(
                input_text, return_tensors='pt', padding=True).input_ids
            gen_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                          top_k=0.0, top_p=1.0,
                                          do_sample=True)
            response_token_ids = model.generate(
                input_ids=input_ids, generation_config=gen_config)
            generated_text = tokenizer.decode(
                response_token_ids[0], skip_special_tokens=True)
            toxic_score = toxic_eval.compute(
                predictions=[(input_text + " " + generated_text)])
            toxics.extend(toxic_score['toxicity'])

        mean = np.mean(toxics)
        std = np.std(toxics)

        return mean, std
