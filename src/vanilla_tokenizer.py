import json
import numpy as np
import pandas as pd
from tqdm import tqdm


INPUT_CSV = "data/processed/swissprot_clean.csv"

OUTPUT_NPY = "data/processed/vanilla_tokenized_sequences.npy"
VOCAB_PATH = "data/processed/vanilla_vocab.json"


print("Loading Swiss-Prot dataset...")
df = pd.read_csv(INPUT_CSV)

print(f"Loaded {len(df)} sequences")



# Amino acids


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")



# Special tokens


SPECIAL_TOKENS = [
    "<PAD>",
    "<START>",
    "<END>",
    "<UNK>",
]



# Vocabulary


VOCAB = SPECIAL_TOKENS + AMINO_ACIDS

token_to_id = {
    token: idx
    for idx, token in enumerate(VOCAB)
}

id_to_token = {
    idx: token
    for token, idx in token_to_id.items()
}

print(f"Vocabulary size: {len(VOCAB)}")



# Encode sequence


def encode_sequence(sequence, max_length=512):

    tokens = ["<START>"] + list(sequence) + ["<END>"]

    token_ids = [
        token_to_id.get(
            token,
            token_to_id["<UNK>"]
        )
        for token in tokens
    ]

    if len(token_ids) < max_length:
        token_ids += [
            token_to_id["<PAD>"]
        ] * (max_length - len(token_ids))

    else:
        token_ids = token_ids[:max_length]

    return token_ids



# Tokenize all proteins


MAX_LENGTH = 512

encoded_sequences = []

for sequence in tqdm(df["sequence"]):

    encoded = encode_sequence(
        sequence,
        MAX_LENGTH
    )

    encoded_sequences.append(encoded)



# Convert to numpy


X = np.array(
    encoded_sequences,
    dtype=np.int32
)

print("\nShape:", X.shape)



# Save


np.save(
    OUTPUT_NPY,
    X
)

with open(VOCAB_PATH, "w") as f:
    json.dump(
        token_to_id,
        f,
        indent=4
    )

print("\nSaved:")
print(OUTPUT_NPY)
print(VOCAB_PATH)