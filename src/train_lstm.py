import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm

from vanilla_dataset import create_dataloaders
from lstm_model import ProteinLSTM


VOCAB_PATH = "data/processed/vanilla_vocab.json"

MODEL_DIR = "models"
MODEL_PATH = "models/protein_lstm_baseline.pt"


BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-4

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.1


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


with open(VOCAB_PATH, "r") as f:
    token_to_id = json.load(f)

vocab_size = len(token_to_id)
pad_token_id = token_to_id["<PAD>"]

print(f"Vocabulary size: {vocab_size}")


train_loader, val_loader = create_dataloaders(
    batch_size=BATCH_SIZE,
    train_split=0.9,
)


model = ProteinLSTM(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)

model = model.to(device)


criterion = nn.CrossEntropyLoss(
    ignore_index=pad_token_id
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


def train_one_epoch():
    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )

    for input_ids, target_ids in progress_bar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=loss.item()
        )

    return total_loss / len(train_loader)


def validate():
    model.eval()

    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(
            val_loader,
            desc="Validation"
        )

        for input_ids, target_ids in progress_bar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch()
        val_loss = validate()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "token_to_id": token_to_id,
                    "config": {
                        "vocab_size": vocab_size,
                        "embed_dim": EMBED_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                    },
                },
                MODEL_PATH,
            )

            print(f"Saved best LSTM model to {MODEL_PATH}")

    print("\nLSTM training complete.")


if __name__ == "__main__":
    main()