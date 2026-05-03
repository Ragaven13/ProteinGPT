import os
import pandas as pd


INPUT_CSV = "data/processed/swissprot_clean.csv"
OUTPUT_CSV = "data/processed/swissprot_labeled.csv"


def assign_function_label(protein_name):
    """
    Assign a broad functional class using keyword matching.

    This is a weak-labeling strategy:
    - It is not perfect.
    - But it is very useful for building the first conditional generation dataset.
    """

    name = str(protein_name).lower()

    if "kinase" in name or "enzyme" in name or "hydrolase" in name or "transferase" in name:
        return "enzyme"

    elif "transporter" in name or "channel" in name or "carrier" in name:
        return "transporter"

    elif "receptor" in name:
        return "receptor"

    elif "dna-binding" in name or "dna binding" in name or "transcription factor" in name:
        return "dna_binding"

    elif "membrane" in name:
        return "membrane"

    elif "antibody" in name or "immunoglobulin" in name or "immune" in name:
        return "immune"

    else:
        return "other"


def main():
    print("Loading cleaned Swiss-Prot dataset...")

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    print(f"Loaded sequences: {len(df)}")

    print("Assigning function labels...")
    df["function_label"] = df["protein_name"].apply(assign_function_label)

    print("\nLabel distribution:")
    print(df["function_label"].value_counts())

    df_labeled = df[df["function_label"] != "other"].copy()

    print(f"\nSequences after removing 'other': {len(df_labeled)}")

    df_labeled.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved labeled dataset to: {OUTPUT_CSV}")
    print(df_labeled.head())


if __name__ == "__main__":
    main()