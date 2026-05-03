import os
import pandas as pd
from Bio import SeqIO


RAW_FASTA_PATH = "data/raw/uniprot_sprot.fasta"
PROCESSED_DIR = "data/processed"
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "swissprot_clean.csv")


def parse_swissprot_fasta(fasta_path):
    records = []

    print(f"Reading FASTA file from: {fasta_path}")

    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        header = record.description
        sequence = str(record.seq)

        protein_id = record.id.split("|")[1] if "|" in record.id else record.id

        protein_name = header.split("OS=")[0]
        protein_name = protein_name.replace(record.id, "").strip()

        organism = ""
        if "OS=" in header:
            organism = header.split("OS=")[1].split("OX=")[0].strip()

        records.append({
            "protein_id": protein_id,
            "protein_name": protein_name,
            "organism": organism,
            "sequence": sequence,
            "sequence_length": len(sequence)
        })

        if i < 3:
            print(f"Example {i+1}: {protein_id}, length={len(sequence)}")

    return pd.DataFrame(records)


def clean_sequences(df):
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    before = len(df)

    df = df.drop_duplicates(subset=["sequence"])
    df = df[df["sequence"].apply(lambda x: set(x).issubset(valid_amino_acids))]
    df = df[df["sequence_length"].between(30, 1000)]

    after = len(df)

    print(f"Removed sequences: {before - after}")

    return df.reset_index(drop=True)


def main():
    print("Starting Swiss-Prot preprocessing...")

    if not os.path.exists(RAW_FASTA_PATH):
        print(f"ERROR: FASTA file not found at: {RAW_FASTA_PATH}")
        print("Check your actual filename using:")
        print("ls data/raw")
        return

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = parse_swissprot_fasta(RAW_FASTA_PATH)

    print(f"Raw sequences: {len(df)}")

    if len(df) == 0:
        print("ERROR: No sequences found. Check if the FASTA file is correct.")
        return

    df_clean = clean_sequences(df)

    print(f"Clean sequences: {len(df_clean)}")

    df_clean.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved processed dataset to: {OUTPUT_CSV}")
    print(df_clean.head())


if __name__ == "__main__":
    main()