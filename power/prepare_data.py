"""
Unified data preparation script for math RL training.
Generates both test (validation) and training datasets with decontamination.

Usage:
    python3 prepare_data.py                  # Test data only
    python3 prepare_data.py --train simplerl # Test + simplerl training data
    python3 prepare_data.py --train dapo     # Test + dapo training data  
    python3 prepare_data.py --train openr1   # Test + openr1 training data
    python3 prepare_data.py --force-test     # Force regenerate test data
"""

import argparse
import os
import pandas as pd
import re
from datasets import load_dataset
from tqdm import tqdm

TEST_DATA_FILE = "test.parquet"

# ================= CONFIGURATION =================
# N-gram decontamination settings
NGRAM_SIZE = 10          # Size of n-grams (10 is standard for decontamination)
OVERLAP_THRESHOLD = 0.5  # Jaccard similarity threshold (0.5 = 50% overlap)

# Qwen-Math Native Template (expects \boxed{} output)
QWEN_TEMPLATE = (
    "<|im_start|>system\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>user\n"
    "{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<|im_start|>think\n"
)

# Test dataset configurations: (hf_path, split, question_key, answer_key, data_source, filter_fn)
TEST_DATASETS = [
    ("HuggingFaceH4/MATH-500", "test", "problem", "answer", "math500", None),
    ("HuggingFaceH4/aime_2024", "train", "problem", "answer", "aime24", None),
    ("GY2233/AIME-2024-2025", "train", "Problem", "Answer", "aime25", 
     lambda x: x.get('ID', '').startswith('aime25')),
    ("math-ai/amc23", "test", "question", "answer", "amc23", None),
    ("lmms-lab/OlympiadBench", "test_en", "question", "final_answer", "olympiad_bench",
     lambda x: x['subfield'] in {'Algebra', 'Geometry', 'Combinatorics', 'Number Theory'} 
               and x['final_answer'] is not None),
    ("math-ai/minervamath", "test", "question", "answer", "minerva", None),
]

# Training dataset configurations: (hf_path, split, question_key, answer_key)
TRAIN_DATASETS = {
    "simplerl": ("zwhe99/simplerl", "train", "problem", "answer"),
    "dapo": ("open-r1/DAPO-Math-17k-Processed", "train", "prompt", "solution"),
    "openr1": ("open-r1/OpenR1-Math-220k", "train", "problem", "answer"),
}

# =================================================

def clean_answer(a):
    """Clean answer formats from various datasets."""
    # Handle OlympiadBench list format: ['$\\frac{1}{2}$'] → \\frac{1}{2}
    if isinstance(a, list):
        a = a[0] if a else ""
    a = str(a)
    # Remove $ delimiters if present
    a = a.strip('$').strip()
    return a


def normalize_question(text):
    """Normalize question text for robust matching."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[\$\\{}\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_ngrams(text, n=NGRAM_SIZE):
    """Extract character n-grams from text."""
    text = normalize_question(text)
    if len(text) < n:
        return set([text]) if text else set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def ngram_overlap(ngrams1, ngrams2):
    """Compute Jaccard similarity between two n-gram sets."""
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0


def make_verl_format(question, answer, data_source, split="test"):
    """Create VeRL-compatible sample format."""
    answer = clean_answer(answer)
    return {
        "prompt": [{"role": "user", "content": QWEN_TEMPLATE.format(question=question)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "data_source": data_source,
        "extra_info": {"split": split, "original_question": question}
    }


def prepare_test_data():
    """Load and format all test/validation datasets."""
    print("\n" + "="*60)
    print("PREPARING TEST DATA")
    print("="*60)
    
    all_samples = []
    
    for hf_path, split, q_key, a_key, data_source, filter_fn in TEST_DATASETS:
        print(f"\nProcessing {data_source} ({hf_path})...")
        ds = load_dataset(hf_path, split=split)
        
        # Apply filter if provided
        if filter_fn:
            ds = ds.filter(filter_fn)
        
        # Limit olympiad_bench and minerva to 200 samples
        if data_source in ["olympiad_bench", "minerva"]:
            ds = ds.select(range(min(200, len(ds))))
        
        # Convert to VeRL format
        for ex in ds:
            sample = make_verl_format(ex[q_key], ex[a_key], data_source, "test")
            all_samples.append(sample)
        
        print(f"  Added {len(ds)} samples")
    
    # Save
    df = pd.DataFrame(all_samples)
    df.to_parquet(TEST_DATA_FILE)
    print(f"\n✅ Saved {len(df)} test samples to '{TEST_DATA_FILE}'")
    
    return all_samples


def load_existing_test_data():
    """Load test samples from existing parquet file."""
    print("\n" + "="*60)
    print(f"LOADING EXISTING TEST DATA: {TEST_DATA_FILE}")
    print("="*60)
    
    df = pd.read_parquet(TEST_DATA_FILE)
    samples = df.to_dict('records')
    print(f"✅ Loaded {len(samples)} test samples")
    return samples


def prepare_train_data(dataset_choice, test_samples):
    """Load, decontaminate, and format training data."""
    print("\n" + "="*60)
    print(f"PREPARING TRAINING DATA: {dataset_choice.upper()}")
    print("="*60)
    
    if dataset_choice not in TRAIN_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_choice}. Choose from: {list(TRAIN_DATASETS.keys())}")
    
    hf_path, split, q_key, a_key = TRAIN_DATASETS[dataset_choice]
    
    # Build decontamination blocklist from test samples
    print("\nBuilding decontamination blocklist...")
    test_questions = [s['extra_info']['original_question'] for s in test_samples]
    exact_set = set(normalize_question(q) for q in test_questions if q)
    exact_set.discard('')
    ngram_sets = [get_ngrams(q) for q in test_questions if q]
    ngram_sets = [ng for ng in ngram_sets if ng]
    print(f"  {len(exact_set)} exact patterns, {len(ngram_sets)} n-gram sets")
    
    # Load training data
    print(f"\nLoading {hf_path}...")
    ds = load_dataset(hf_path, split=split)
    print(f"  Loaded {len(ds)} raw samples")
    
    # Process with decontamination
    clean_samples = []
    exact_matches = 0
    ngram_matches = 0
    
    for ex in tqdm(ds, desc="Decontaminating"):
        q = ex.get(q_key)
        a = ex.get(a_key)
        
        if not q or not a:
            continue
        
        # Exact match check
        normalized = normalize_question(q)
        if normalized in exact_set:
            exact_matches += 1
            continue
        
        # N-gram overlap check
        q_ngrams = get_ngrams(q)
        is_contaminated = False
        if q_ngrams:
            for val_ngrams in ngram_sets:
                if ngram_overlap(q_ngrams, val_ngrams) >= OVERLAP_THRESHOLD:
                    is_contaminated = True
                    break
        
        if is_contaminated:
            ngram_matches += 1
            continue
        
        # Add clean sample
        sample = make_verl_format(q, a, dataset_choice, "train")
        clean_samples.append(sample)
    
    # Save
    output_file = f"train_{dataset_choice}.parquet"
    df = pd.DataFrame(clean_samples)
    df.to_parquet(output_file)
    
    print(f"\n{'='*40}")
    print(f"DONE!")
    print(f"Original Size: {len(ds)}")
    print(f"Removed (Contaminated): {exact_matches + ngram_matches}")
    print(f"  - Exact matches: {exact_matches}")
    print(f"  - N-gram matches: {ngram_matches}")
    print(f"Final Size: {len(df)}")
    print(f"✅ Saved to: {output_file}")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description="Prepare math RL training/test data")
    parser.add_argument("--train", type=str, choices=["simplerl", "dapo", "openr1"],
                        help="Training dataset to prepare")
    parser.add_argument("--force-test", action="store_true",
                        help="Force regenerate test data even if file exists")
    args = parser.parse_args()
    
    # Smart default: load existing test data if available, generate if not
    test_file_exists = os.path.exists(TEST_DATA_FILE)
    
    if args.force_test or not test_file_exists:
        test_samples = prepare_test_data()
    else:
        test_samples = load_existing_test_data()
    
    # Generate training data if requested
    if args.train:
        prepare_train_data(args.train, test_samples)
    
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()

