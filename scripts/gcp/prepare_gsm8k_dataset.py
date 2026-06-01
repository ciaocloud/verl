#!/usr/bin/env python3
"""Prepare GSM8K parquet files in verl's RLHF dataset schema."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import load_dataset


def extract_solution(solution_str: str) -> str:
    match = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if match is None:
        raise ValueError(f"Could not extract GSM8K final answer from: {solution_str}")
    return match.group(1).replace(",", "")


def convert_split(dataset, split: str):
    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def process(example, idx):
        question_raw = example["question"]
        answer_raw = example["answer"]
        return {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": f"{question_raw} {instruction}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": extract_solution(answer_raw)},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }

    return dataset.map(process, with_indices=True, remove_columns=dataset.column_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Directory for train.parquet and test.parquet.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional row limit for smoke datasets.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional row limit for smoke datasets.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("openai/gsm8k", "main")
    train = dataset["train"]
    test = dataset["test"]

    if args.train_limit is not None:
        train = train.select(range(min(args.train_limit, len(train))))
    if args.test_limit is not None:
        test = test.select(range(min(args.test_limit, len(test))))

    convert_split(train, "train").to_parquet(output_dir / "train.parquet")
    convert_split(test, "test").to_parquet(output_dir / "test.parquet")

    print(f"Wrote {output_dir / 'train.parquet'}")
    print(f"Wrote {output_dir / 'test.parquet'}")


if __name__ == "__main__":
    main()
