import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Load dataset
DATA_PATH = "data/processed/swissprot_clean.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} sequences")

# Define amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Special tokens
SPECIAL_TOKENS = ["<PAD>", "<START>", "<END>", "<UNK>"]

# Build vocabulary
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
token_to_id = {token: idx for idx, token in enumerate(VOCAB)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

# Encoding function
def encode_sequence(seq, max_length=512):
    tokens = ["<START>"] + list(seq) + ["<END>"]

    token_ids = [
        token_to_id.get(token, token_to_id["<UNK>"])
        for token in tokens
    ]

    # Padding / truncation
    if len(token_ids) < max_length:
        token_ids += [token_to_id["<PAD>"]] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]

    return token_ids

# Apply encoding
MAX_LENGTH = 512
encoded_sequences = []

for seq in tqdm(df["sequence"]):
    encoded = encode_sequence(seq, MAX_LENGTH)
    encoded_sequences.append(encoded)

# Convert to numpy array
X = np.array(encoded_sequences)
print("Shape:", X.shape)

# Save tokenized sequences
np.save("data/processed/tokenized_sequences.npy", X)

# Save vocabulary
with open("data/processed/vocab.json", "w") as f:
    json.dump(token_to_id, f)

print("Tokenization complete. Files saved in data/processed/")