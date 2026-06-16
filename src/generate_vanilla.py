import os
import torch
import pandas as pd

from model import ProteinGPT


MODEL_PATH = "models/proteingpt_vanilla.pt"
RESULTS_DIR = "results"
OUTPUT_CSV = "results/generated_vanilla_500.csv"

NUM_SEQUENCES = 500


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    token_to_id = checkpoint["token_to_id"]
    config = checkpoint["config"]

    id_to_token = {
        idx: token
        for token, idx in token_to_id.items()
    }

    model = ProteinGPT(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, token_to_id, id_to_token


def generate_sequence(
    model,
    token_to_id,
    id_to_token,
    max_new_tokens=300,
    temperature=0.9,
):
    start_token = token_to_id["<START>"]
    end_token = token_to_id["<END>"]

    generated = [start_token]

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long)

        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[0, -1, :] / temperature
        probabilities = torch.softmax(next_token_logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        if next_token == end_token:
            break

        generated.append(next_token)

    tokens = [
        id_to_token[token_id]
        for token_id in generated
    ]

    amino_acids = [
        token
        for token in tokens
        if token not in [
            "<START>",
            "<END>",
            "<PAD>",
            "<UNK>",
        ]
    ]

    return "".join(amino_acids)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, token_to_id, id_to_token = load_model(MODEL_PATH)

    generated_records = []

    print(f"Generating {NUM_SEQUENCES} vanilla protein sequences...")

    for i in range(NUM_SEQUENCES):
        sequence = generate_sequence(
            model=model,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            max_new_tokens=300,
            temperature=0.9,
        )

        generated_records.append({
            "model": "vanilla_gpt",
            "generated_sequence": sequence,
            "sequence_length": len(sequence),
        })

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{NUM_SEQUENCES}")

    df = pd.DataFrame(generated_records)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(f"\nSaved {len(df)} vanilla sequences to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()