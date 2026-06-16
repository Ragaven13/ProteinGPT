import numpy as np
from difflib import SequenceMatcher


def sequence_similarity(seq1, seq2):
    """
    Calculates similarity between two protein sequences.

    Output range:
    0.0 = completely different
    1.0 = identical
    """
    return SequenceMatcher(None, seq1, seq2).ratio()


def amino_acid_distribution(sequence):
    """
    Calculates amino acid frequency distribution.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    total = len(sequence)

    if total == 0:
        return {aa: 0 for aa in amino_acids}

    return {
        aa: sequence.count(aa) / total
        for aa in amino_acids
    }


def diversity_score(sequences):
    """
    Average pairwise difference between generated sequences.

    Higher score = more diverse generation.
    """
    scores = []

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            sim = sequence_similarity(sequences[i], sequences[j])
            scores.append(1 - sim)

    return np.mean(scores) if scores else 0


def novelty_score(generated_sequences, real_sequences):
    """
    Measures how different generated sequences are from real training sequences.

    Higher score = more novel.
    """
    novelty_scores = []

    for gen_seq in generated_sequences:
        max_similarity = max(
            sequence_similarity(gen_seq, real_seq)
            for real_seq in real_sequences[:1000]
        )

        novelty_scores.append(1 - max_similarity)

    return np.mean(novelty_scores)


if __name__ == "__main__":
    generated = [
        "MKTLLAVVG",
        "MKAILVVGG",
        "GHTLLAVVV",
    ]

    real = [
        "MKTLLAVVA",
        "MKLLLLVGG",
        "GGGTTTAAA",
    ]

    print("Similarity example:", sequence_similarity(generated[0], real[0]))
    print("Amino acid distribution:", amino_acid_distribution(generated[0]))
    print("Diversity score:", diversity_score(generated))
    print("Novelty score:", novelty_score(generated, real))