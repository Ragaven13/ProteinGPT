import sys
import torch
import pandas as pd
import streamlit as st

sys.path.append("src")

from model import ProteinGPT


MODEL_PATH = "models/proteingpt_conditional.pt"


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    token_to_id = checkpoint["token_to_id"]
    config = checkpoint["config"]

    id_to_token = {
        idx: token
        for token, idx in token_to_id.items()
    }

    model = ProteinGPT(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, token_to_id, id_to_token


def generate_sequence(
    model,
    token_to_id,
    id_to_token,
    function_label,
    max_new_tokens,
    temperature,
):
    start_token = token_to_id["<START>"]
    function_token = token_to_id[f"<{function_label}>"]
    end_token = token_to_id["<END>"]

    generated = [start_token, function_token]

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long)

        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[0, -1, :] / temperature
        probabilities = torch.softmax(next_token_logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        if next_token == end_token:
            break

        generated.append(next_token)

    tokens = [
        id_to_token[token_id]
        for token_id in generated
    ]

    amino_acids = [
        token
        for token in tokens
        if token not in [
            "<START>",
            "<END>",
            f"<{function_label}>",
            "<PAD>",
            "<UNK>",
        ]
    ]

    return "".join(amino_acids)


def amino_acid_distribution(sequence):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    if len(sequence) == 0:
        return pd.DataFrame({
            "Amino Acid": amino_acids,
            "Frequency": [0] * len(amino_acids)
        })

    return pd.DataFrame({
        "Amino Acid": amino_acids,
        "Frequency": [
            sequence.count(aa) / len(sequence)
            for aa in amino_acids
        ]
    })


def main():
    st.set_page_config(
        page_title="ProteinGPT",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 ProteinGPT")
    st.subheader("Function-Conditioned Protein Sequence Generation")

    st.markdown(
        """
        ProteinGPT is a GPT-style Transformer trained on UniProt Swiss-Prot protein sequences.
        It generates protein-like amino acid sequences conditioned on biological function labels.
        """
    )

    model, token_to_id, id_to_token = load_model()

    st.sidebar.header("Generation Settings")

    function_label = st.sidebar.selectbox(
        "Select Protein Function",
        [
            "enzyme",
            "transporter",
            "membrane",
            "receptor",
            "dna_binding",
        ],
    )

    max_new_tokens = st.sidebar.slider(
        "Maximum Sequence Length",
        min_value=50,
        max_value=500,
        value=300,
        step=50,
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.3,
        max_value=1.5,
        value=0.8,
        step=0.1,
    )

    num_sequences = st.sidebar.slider(
        "Number of Sequences",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    if st.button("Generate Protein Sequences"):
        generated_records = []

        with st.spinner("Generating protein sequences..."):
            for i in range(num_sequences):
                sequence = generate_sequence(
                    model=model,
                    token_to_id=token_to_id,
                    id_to_token=id_to_token,
                    function_label=function_label,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                generated_records.append({
                    "Function": function_label,
                    "Sequence": sequence,
                    "Length": len(sequence),
                })

        df = pd.DataFrame(generated_records)

        st.success("Generation complete!")

        st.subheader("Generated Protein Sequences")

        for idx, row in df.iterrows():
            st.markdown(f"### Sequence {idx + 1}")
            st.write(f"**Function:** {row['Function']}")
            st.write(f"**Length:** {row['Length']}")
            st.code(row["Sequence"], language="text")

        st.subheader("Generated Sequence Table")
        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Generated Sequences as CSV",
            data=csv,
            file_name="generated_proteins.csv",
            mime="text/csv",
        )

        st.subheader("Amino Acid Distribution")

        selected_sequence = df.iloc[0]["Sequence"]
        dist_df = amino_acid_distribution(selected_sequence)

        st.bar_chart(
            dist_df.set_index("Amino Acid")
        )

    st.markdown("---")

    st.markdown(
        """
        ### Models Compared

        | Model | Novelty | Diversity | Function Control |
        |---|---:|---:|---|
        | LSTM Baseline | 0.7229 | 0.9336 | No |
        | Vanilla GPT | 0.7343 | 0.9508 | No |
        | ProteinGPT | **0.7356** | **0.9574** | Yes |
        """
    )


if __name__ == "__main__":
    main()
