import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


 
# 1. File paths


INPUT_CSV = "data/processed/swissprot_labeled.csv"
OUTPUT_NPY = "data/processed/conditional_tokenized_sequences.npy"
VOCAB_PATH = "data/processed/conditional_vocab.json"



# 2. Load labeled dataset


df = pd.read_csv(INPUT_CSV)

print(f"Loaded labeled sequences: {len(df)}")
print("\nFunction label distribution:")
print(df["function_label"].value_counts())



# 3. Define amino acid tokens


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")



# 4. Define special tokens


SPECIAL_TOKENS = [
    "<PAD>",
    "<START>",
    "<END>",
    "<UNK>",
]



# 5. Define function tokens


FUNCTION_TOKENS = [
    "<enzyme>",
    "<transporter>",
    "<membrane>",
    "<receptor>",
    "<dna_binding>",
]



# 6. Build vocabulary

VOCAB = SPECIAL_TOKENS + FUNCTION_TOKENS + AMINO_ACIDS

token_to_id = {token: idx for idx, token in enumerate(VOCAB)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

print("\nVocabulary size:", len(VOCAB))
print("Token mapping example:")
print(token_to_id)



# 7. Encode one sequence


def encode_conditional_sequence(function_label, sequence, max_length=512):
    """
    Convert function label + protein sequence into token IDs.

    Example:
    function_label = "enzyme"
    sequence = "MKT"

    Tokens:
    ["<START>", "<enzyme>", "M", "K", "T", "<END>"]

    IDs:
    [1, 4, 16, 13, 21, 2]
    """

    function_token = f"<{function_label}>"

    tokens = ["<START>", function_token] + list(sequence) + ["<END>"]

    token_ids = [
        token_to_id.get(token, token_to_id["<UNK>"])
        for token in tokens
    ]

    if len(token_ids) < max_length:
        token_ids += [token_to_id["<PAD>"]] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]

    return token_ids



# 8. Filter labels we want


allowed_labels = [
    "enzyme",
    "transporter",
    "membrane",
    "receptor",
    "dna_binding",
]

df = df[df["function_label"].isin(allowed_labels)].copy()

print(f"\nSequences after keeping selected labels: {len(df)}")



# 9. Tokenize all sequences


MAX_LENGTH = 512

encoded_sequences = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    encoded = encode_conditional_sequence(
        function_label=row["function_label"],
        sequence=row["sequence"],
        max_length=MAX_LENGTH,
    )

    encoded_sequences.append(encoded)



# 10. Save tokenized array


X = np.array(encoded_sequences, dtype=np.int32)

print("\nTokenized shape:", X.shape)

np.save(OUTPUT_NPY, X)



# 11. Save vocabulary


os.makedirs("data/processed", exist_ok=True)

with open(VOCAB_PATH, "w") as f:
    json.dump(token_to_id, f, indent=4)

print(f"\nSaved tokenized data to: {OUTPUT_NPY}")
print(f"Saved vocabulary to: {VOCAB_PATH}")
print("\nConditional tokenization complete.")