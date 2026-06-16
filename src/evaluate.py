import pandas as pd
import numpy as np
from difflib import SequenceMatcher


GENERATED_CSV = "results/generated_sequences.csv"
REAL_CSV = "data/processed/swissprot_labeled.csv"
OUTPUT_CSV = "results/evaluation_results.csv"


def sequence_similarity(seq1, seq2):
    return SequenceMatcher(None, seq1, seq2).ratio()


def diversity_score(sequences):
    scores = []

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            sim = sequence_similarity(sequences[i], sequences[j])
            scores.append(1 - sim)

    return np.mean(scores) if scores else 0


def novelty_score(generated_sequences, real_sequences, sample_size=1000):
    real_sample = real_sequences[:sample_size]

    scores = []

    for gen_seq in generated_sequences:
        max_similarity = max(
            sequence_similarity(gen_seq, real_seq)
            for real_seq in real_sample
        )

        scores.append(1 - max_similarity)

    return np.mean(scores)


def average_length(sequences):
    return np.mean([len(seq) for seq in sequences])


def main():
    print("Loading generated sequences...")
    generated_df = pd.read_csv(GENERATED_CSV)

    print("Loading real Swiss-Prot labeled sequences...")
    real_df = pd.read_csv(REAL_CSV)

    results = []

    for label in generated_df["function_label"].unique():
        print(f"\nEvaluating label: {label}")

        generated_sequences = generated_df[
            generated_df["function_label"] == label
        ]["generated_sequence"].dropna().tolist()

        real_sequences = real_df[
            real_df["function_label"] == label
        ]["sequence"].dropna().tolist()

        if len(generated_sequences) == 0 or len(real_sequences) == 0:
            continue

        novelty = novelty_score(generated_sequences, real_sequences)
        diversity = diversity_score(generated_sequences)
        gen_avg_len = average_length(generated_sequences)
        real_avg_len = average_length(real_sequences[:1000])

        results.append({
            "function_label": label,
            "num_generated": len(generated_sequences),
            "num_real_compared": min(1000, len(real_sequences)),
            "novelty_score": novelty,
            "diversity_score": diversity,
            "generated_avg_length": gen_avg_len,
            "real_avg_length_sample": real_avg_len,
        })

    results_df = pd.DataFrame(results)

    print("\nEvaluation Results:")
    print(results_df)

    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved evaluation results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()