import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "results/model_comparison.csv"
OUTPUT_DIR = "results/figures"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # Novelty plot
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["novelty_score"])
    plt.title("Novelty Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("Novelty Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/novelty_comparison.png", dpi=300)
    plt.close()

    # Diversity plot
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["diversity_score"])
    plt.title("Diversity Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("Diversity Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/diversity_comparison.png", dpi=300)
    plt.close()

    # Average length plot
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["generated_avg_length"])
    plt.title("Generated Sequence Length Comparison")
    plt.xlabel("Model")
    plt.ylabel("Average Sequence Length")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/length_comparison.png", dpi=300)
    plt.close()

    print("Saved plots to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()