import json
import torch
from model import ProteinGPT


MODEL_PATH = "models/proteingpt_conditional.pt"


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    token_to_id = checkpoint["token_to_id"]
    config = checkpoint["config"]

    id_to_token = {idx: token for token, idx in token_to_id.items()}

    model = ProteinGPT(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, token_to_id, id_to_token


def generate_sequence(
    model,
    token_to_id,
    id_to_token,
    function_label="enzyme",
    max_new_tokens=300,
    temperature=1.0,
):
    start_token = token_to_id["<START>"]
    function_token = token_to_id[f"<{function_label}>"]
    end_token = token_to_id["<END>"]

    generated = [start_token, function_token]

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long)

        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[0, -1, :] / temperature
        probabilities = torch.softmax(next_token_logits, dim=-1)

        next_token = torch.multinomial(probabilities, num_samples=1).item()

        if next_token == end_token:
            break

        generated.append(next_token)

    tokens = [id_to_token[token_id] for token_id in generated]

    amino_acids = [
        token for token in tokens
        if token not in ["<START>", "<END>", f"<{function_label}>", "<PAD>", "<UNK>"]
    ]

    return "".join(amino_acids)


def main():
    model, token_to_id, id_to_token = load_model(MODEL_PATH)

    function_labels = [
        "enzyme",
        "transporter",
        "membrane",
        "receptor",
        "dna_binding",
    ]

    for label in function_labels:
        print(f"\nGenerated {label} protein:")
        sequence = generate_sequence(
            model=model,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            function_label=label,
            max_new_tokens=300,
            temperature=0.8,
        )
        print(sequence)


if __name__ == "__main__":
    main()