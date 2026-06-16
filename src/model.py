import torch
import torch.nn as nn


class ProteinGPT(nn.Module):
    """
    GPT-style Transformer model for protein sequence generation.

    Input:
        token IDs with shape: [batch_size, sequence_length]

    Output:
        logits with shape: [batch_size, sequence_length, vocab_size]
    """

    def __init__(
        self,
        vocab_size,
        max_length=511,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim

        # Converts token IDs into dense vectors
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )

        # Adds position information: token 1, token 2, token 3, etc.
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim,
        )

        # One Transformer decoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        # Stack multiple transformer layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Final layer: converts hidden vectors back to vocabulary scores
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def create_causal_mask(self, seq_len, device):
        """
        Creates a mask so the model cannot see future tokens.

        Example:
        When predicting token 3, model can only see tokens 1 and 2.
        """

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        )

        mask = mask.masked_fill(mask == 1, float("-inf"))

        return mask

    def forward(self, input_ids):
        """
        Forward pass.

        input_ids shape:
            [batch_size, seq_len]

        logits shape:
            [batch_size, seq_len, vocab_size]
        """

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token positions: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device)

        # Add batch dimension: [1, seq_len]
        positions = positions.unsqueeze(0)

        # Token embeddings
        token_emb = self.token_embedding(input_ids)

        # Position embeddings
        pos_emb = self.position_embedding(positions)

        # Combine token meaning + position meaning
        x = token_emb + pos_emb

        # Causal mask prevents cheating
        causal_mask = self.create_causal_mask(seq_len, device)

        # Transformer processing
        x = self.transformer(x, mask=causal_mask)

        # Convert hidden vectors to next-token scores
        logits = self.output_head(x)

        return logits


if __name__ == "__main__":
    vocab_size = 29

    model = ProteinGPT(
        vocab_size=vocab_size,
        max_length=511,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
    )

    dummy_input = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 511),
    )

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)