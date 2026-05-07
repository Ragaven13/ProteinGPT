import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


TOKENIZED_PATH = "data/processed/conditional_tokenized_sequences.npy"


class ProteinSequenceDataset(Dataset):
    """
    PyTorch Dataset for ProteinGPT.

    It loads tokenized protein sequences and creates:
    - input_ids: all tokens except the last one
    - target_ids: all tokens except the first one

    This is how GPT learns next-token prediction.
    """

    def __init__(self, tokenized_path):
        self.data = np.load(tokenized_path)
        print(f"Loaded tokenized data with shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]

        input_ids = sequence[:-1]
        target_ids = sequence[1:]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        return input_ids, target_ids


def create_dataloaders(batch_size=32, train_split=0.9):
    dataset = ProteinSequenceDataset(TOKENIZED_PATH)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders(batch_size=4)

    for input_ids, target_ids in train_loader:
        print("Input shape:", input_ids.shape)
        print("Target shape:", target_ids.shape)
        print("Example input:", input_ids[0][:20])
        print("Example target:", target_ids[0][:20])
        break