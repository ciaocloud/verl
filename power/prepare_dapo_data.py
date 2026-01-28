#!/usr/bin/env python3
"""
Prepare DAPO-Math-17k and validation datasets for verl training.

Usage:
    python prepare_dapo_data.py                              # All datasets with default split
    python prepare_dapo_data.py --dataset dapo               # DAPO only
    python prepare_dapo_data.py --dataset math500            # MATH-500 only
    python prepare_dapo_data.py --dataset aime               # AIME-2024 only
    python prepare_dapo_data.py --split_val 0                # No split (all 17k for training)
    python prepare_dapo_data.py --template qwen_math         # Qwen-Math style
    python prepare_dapo_data.py --output_dir ./data          # Custom base output dir
"""

import argparse
import os

from datasets import load_dataset


# Template definitions
TEMPLATES = {
    "original": (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form Answer: $Answer (without quotes) "
        "where $Answer is the answer to the problem.\n\n"
        "{question}\n\n"
        'Remember to put your answer on its own line after "Answer:".'
    ),
    "boxed": (
        "Solve the following math problem step by step. "
        "Put your final answer within \\boxed{{}}.\n\n"
        "{question}"
    ),
    "qwen_math": (
        "{question}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}."
    ),
    "r1": (
        "{question}\n\n"
        "Please think step by step and put your final answer within \\boxed{{}}."
    ),
}


def format_prompt(question: str, template: str) -> list[dict]:
    """Format a question into a chat prompt."""
    content = TEMPLATES[template].format(question=question)
    return [{"role": "user", "content": content}]


def prepare_dapo(output_dir: str, template: str, split_val: int = 0):
    """Prepare DAPO-Math-17k training dataset, optionally splitting off validation."""
    print("Loading open-r1/DAPO-Math-17k-Processed...")
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "all")
    dapo_data = ds["train"]
    print(f"Loaded {len(dapo_data):,} samples")

    print("Converting to verl format...")

    def convert(example):
        return {
            "data_source": example["data_source"],
            "prompt": format_prompt(example["prompt"], template),
            "ability": example["ability"],
            "reward_model": example["reward_model"],
            "extra_info": example["extra_info"],
        }

    dapo_verl = dapo_data.map(convert, remove_columns=dapo_data.column_names, desc="Converting")

    if split_val > 0:
        # Split into train and val
        print(f"Splitting: {len(dapo_verl) - split_val:,} train, {split_val} val...")
        dapo_verl = dapo_verl.shuffle(seed=42)
        dapo_split = dapo_verl.train_test_split(test_size=split_val, seed=42)
        
        train_path = os.path.join(output_dir, "train.parquet")
        val_path = os.path.join(output_dir, "test.parquet")
        
        dapo_split["train"].to_parquet(train_path)
        dapo_split["test"].to_parquet(val_path)
        
        train_size = os.path.getsize(train_path) / (1024 * 1024)
        val_size = os.path.getsize(val_path) / 1024
        print(f"✓ {train_path} ({train_size:.1f} MB, {len(dapo_split['train']):,} samples)")
        print(f"✓ {val_path} ({val_size:.1f} KB, {len(dapo_split['test']):,} samples)")
        return train_path, val_path
    else:
        output_path = os.path.join(output_dir, "dapo-math-17k.parquet")
        print(f"Saving to {output_path}...")
        dapo_verl.to_parquet(output_path)

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ {output_path} ({file_size:.1f} MB, {len(dapo_verl):,} samples)")
        return output_path


def prepare_math500(output_dir: str, template: str):
    """Prepare MATH-500 validation dataset."""
    print("Loading HuggingFaceH4/MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500")
    math500_data = ds["test"]
    print(f"Loaded {len(math500_data)} samples")

    print("Converting to verl format...")

    def convert(example, idx):
        return {
            "data_source": "math_dapo",  # Use same as training for consistent reward function
            "prompt": format_prompt(example["problem"], template),
            "ability": example["subject"],
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {"index": idx, "level": example["level"], "unique_id": example["unique_id"]},
        }

    math500_verl = math500_data.map(convert, with_indices=True, remove_columns=math500_data.column_names, desc="Converting")

    output_path = os.path.join(output_dir, "math500.parquet")
    print(f"Saving to {output_path}...")
    math500_verl.to_parquet(output_path)

    file_size = os.path.getsize(output_path) / 1024
    print(f"✓ {output_path} ({file_size:.1f} KB, {len(math500_verl)} samples)")
    return output_path


def prepare_aime(output_dir: str, template: str):
    """Prepare AIME-2024 validation dataset."""
    print("Loading BytedTsinghua-SIA/AIME-2024...")
    ds = load_dataset("BytedTsinghua-SIA/AIME-2024")
    aime_data = ds["train"]
    print(f"Loaded {len(aime_data)} samples (already in verl format)")

    # Remove the extra index column if present
    if "__index_level_0__" in aime_data.column_names:
        aime_data = aime_data.remove_columns(["__index_level_0__"])

    output_path = os.path.join(output_dir, "aime-2024.parquet")
    print(f"Saving to {output_path}...")
    aime_data.to_parquet(output_path)

    file_size = os.path.getsize(output_path) / 1024
    print(f"✓ {output_path} ({file_size:.1f} KB, {len(aime_data)} samples)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DAPO-Math-17k and validation datasets for verl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  dapo     - DAPO-Math-17k training data (17,398 samples)
  math500  - MATH-500 validation (500 samples, medium difficulty)
  aime     - AIME-2024 validation (30 samples, very hard)

Templates:
  original   - "Answer: X" format (DAPO paper default)
  boxed      - Simple "\\boxed{X}" format  
  qwen_math  - Qwen-Math style with \\boxed{}
  r1         - R1/DeepSeek style with \\boxed{}
        """,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/data",
        help="Base output directory (default: /workspace/data)",
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=list(TEMPLATES.keys()),
        default="original",
        help="Prompt template (default: original)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dapo", "math500", "aime", "all"],
        default="all",
        help="Dataset to prepare (default: all)",
    )
    parser.add_argument(
        "--split_val",
        type=int,
        default=500,
        help="Hold out N samples from DAPO for validation (default: 500, 0 for no split)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Preparation for verl")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Template: {args.template}")
    if args.split_val > 0:
        print(f"Split val: {args.split_val} samples from DAPO")
    print(f"Output base: {args.output_dir}")
    print()

    if args.dataset in ["dapo", "all"]:
        dapo_dir = os.path.join(args.output_dir, "dapo")
        os.makedirs(dapo_dir, exist_ok=True)
        prepare_dapo(dapo_dir, args.template, args.split_val)
        print()

    if args.dataset in ["math500", "all"]:
        math500_dir = os.path.join(args.output_dir, "math500")
        os.makedirs(math500_dir, exist_ok=True)
        prepare_math500(math500_dir, args.template)
        print()

    if args.dataset in ["aime", "all"]:
        aime_dir = os.path.join(args.output_dir, "aime")
        os.makedirs(aime_dir, exist_ok=True)
        prepare_aime(aime_dir, args.template)
        print()

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
