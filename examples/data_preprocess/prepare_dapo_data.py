#!/usr/bin/env python3
"""
Prepare DAPO-Math-17k dataset for verl training.

Downloads open-r1/DAPO-Math-17k-Processed and converts to verl format.

Usage:
    python prepare_dapo_data.py                              # Training + validation
    python prepare_dapo_data.py --no_val                     # Training data only
    python prepare_dapo_data.py --template qwen_math         # Qwen-Math style
    python prepare_dapo_data.py --template r1                # R1/DeepSeek style
    python prepare_dapo_data.py --output_dir /workspace/data # Custom output
"""

import argparse
import os
import urllib.request

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

AIME_2024_URL = "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet"


def format_prompt(question: str, template: str) -> list[dict]:
    """Format a question into a chat prompt."""
    content = TEMPLATES[template].format(question=question)
    return [{"role": "user", "content": content}]


def download_file(url: str, output_path: str):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)
    file_size = os.path.getsize(output_path) / 1024
    print(f"Saved to {output_path} ({file_size:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DAPO-Math-17k dataset for verl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Templates:
  original   - "Answer: X" format (DAPO paper default)
  boxed      - Simple "\\boxed{X}" format  
  qwen_math  - Qwen-Math style: concise with \\boxed{}
  r1         - R1/DeepSeek style: "think step by step" with \\boxed{}
        """,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/data",
        help="Output directory (default: /workspace/data)",
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=list(TEMPLATES.keys()),
        default="original",
        help="Prompt template (default: original)",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Skip downloading AIME-2024 validation data",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DAPO-Math-17k Dataset Preparation")
    print("=" * 60)
    print(f"Template: {args.template}")
    print(f"Output: {args.output_dir}")
    print(f"Download validation: {not args.no_val}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and convert training dataset
    print("Loading open-r1/DAPO-Math-17k-Processed...")
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "all")
    train = ds["train"]
    print(f"Loaded {len(train):,} samples")

    print("Converting to verl format...")

    def convert(example):
        return {
            "data_source": example["data_source"],
            "prompt": format_prompt(example["prompt"], args.template),
            "ability": example["ability"],
            "reward_model": example["reward_model"],
            "extra_info": example["extra_info"],
        }

    train_verl = train.map(convert, remove_columns=train.column_names, desc="Converting")

    # Save training data
    train_path = os.path.join(args.output_dir, "dapo-math-17k.parquet")
    print(f"Saving to {train_path}...")
    train_verl.to_parquet(train_path)
    train_size = os.path.getsize(train_path) / (1024 * 1024)

    # Download validation data (default: yes)
    val_path = None
    if not args.no_val:
        print()
        val_path = os.path.join(args.output_dir, "aime-2024.parquet")
        download_file(AIME_2024_URL, val_path)

    # Summary
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Training: {train_path} ({train_size:.1f} MB, {len(train_verl):,} samples)")
    if val_path:
        val_size = os.path.getsize(val_path) / 1024
        print(f"Validation: {val_path} ({val_size:.1f} KB)")
    print()
    print("Sample prompt:")
    print("-" * 40)
    print(train_verl[0]["prompt"][0]["content"][:300] + "...")
    print()
    print(f"Answer: {train_verl[0]['reward_model']['ground_truth']}")


if __name__ == "__main__":
    main()
