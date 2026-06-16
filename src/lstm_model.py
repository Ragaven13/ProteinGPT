import torch
import torch.nn as nn


class ProteinLSTM(nn.Module):
    """
    LSTM baseline for protein sequence generation.

    Input:
        token IDs: [batch_size, sequence_length]

    Output:
        logits: [batch_size, sequence_length, vocab_size]
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.output_head = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        lstm_output, _ = self.lstm(x)

        logits = self.output_head(lstm_output)

        return logits


if __name__ == "__main__":
    vocab_size = 24

    model = ProteinLSTM(vocab_size=vocab_size)

    dummy_input = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 511),
    )

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)