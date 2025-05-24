from gru_model import train_gru
from mlp_model import train_mlp
from lstm_model import train_lstm
from transformer_model import train_transformer

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a volatility model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["GRU", "MLP", "LSTM", "TRANSFORMER"],
        default="GRU",
        help="Model type to train (GRU, MLP, LSTM or TRANSFORMER)"
    )

    args = parser.parse_args()

    if args.model == "GRU":
        train_gru()
    elif args.model == "MLP":
        train_mlp()
    elif args.model == "LSTM":
        train_lstm()
    elif args.model == "TRANSFORMER":
        train_transformer()
    else:
        print("Unsupported model type.")
