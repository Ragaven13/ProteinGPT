import os
import torch
import pandas as pd

from model import ProteinGPT
from generate import load_model, generate_sequence


MODEL_PATH = "models/proteingpt_conditional.pt"
RESULTS_DIR = "results"
OUTPUT_CSV = "results/generated_sequences_500.csv"

NUM_PER_LABEL = 100


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, token_to_id, id_to_token = load_model(MODEL_PATH)

    function_labels = [
        "enzyme",
        "transporter",
        "membrane",
        "receptor",
        "dna_binding",
    ]

    generated_records = []

    for label in function_labels:
        print(f"\nGenerating {NUM_PER_LABEL} {label} proteins...")

        for i in range(NUM_PER_LABEL):
            sequence = generate_sequence(
                model=model,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
                function_label=label,
                max_new_tokens=300,
                temperature=0.9,
            )

            generated_records.append({
                "function_label": label,
                "generated_sequence": sequence,
                "sequence_length": len(sequence),
            })

            if (i + 1) % 10 == 0:
                print(f"{label}: generated {i + 1}/{NUM_PER_LABEL}")

    df = pd.DataFrame(generated_records)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} generated sequences to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()