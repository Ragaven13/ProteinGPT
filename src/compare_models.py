import pandas as pd


CONDITIONAL_CSV = "results/evaluation_results.csv"
VANILLA_CSV = "results/evaluation_vanilla_results.csv"
LSTM_CSV = "results/evaluation_lstm_results.csv"

OUTPUT_CSV = "results/model_comparison.csv"


def main():
    conditional_df = pd.read_csv(CONDITIONAL_CSV)
    vanilla_df = pd.read_csv(VANILLA_CSV)
    lstm_df = pd.read_csv(LSTM_CSV)

    lstm_summary = {
        "model": "lstm_baseline",
        "num_generated": lstm_df["num_generated"].iloc[0],
        "novelty_score": lstm_df["novelty_score"].iloc[0],
        "diversity_score": lstm_df["diversity_score"].iloc[0],
        "generated_avg_length": lstm_df["generated_avg_length"].iloc[0],
        "function_control": "no",
    }

    vanilla_summary = {
        "model": "vanilla_gpt",
        "num_generated": vanilla_df["num_generated"].iloc[0],
        "novelty_score": vanilla_df["novelty_score"].iloc[0],
        "diversity_score": vanilla_df["diversity_score"].iloc[0],
        "generated_avg_length": vanilla_df["generated_avg_length"].iloc[0],
        "function_control": "no",
    }

    conditional_summary = {
        "model": "conditional_gpt",
        "num_generated": conditional_df["num_generated"].sum(),
        "novelty_score": conditional_df["novelty_score"].mean(),
        "diversity_score": conditional_df["diversity_score"].mean(),
        "generated_avg_length": conditional_df["generated_avg_length"].mean(),
        "function_control": "yes",
    }

    comparison_df = pd.DataFrame([
        lstm_summary,
        vanilla_summary,
        conditional_summary,
    ])

    print("\nFinal Model Comparison:")
    print(comparison_df)

    comparison_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved final comparison table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()