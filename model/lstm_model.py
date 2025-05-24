import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import create_engine
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import StandardScaler

uri = os.getenv("MLFLOW_TRACKING_URI", "file:/app/mlruns").replace("file:///", "file:/")
mlflow.set_tracking_uri(uri)
print("👉 Tracking URI:", mlflow.get_tracking_uri())

def train_lstm():
    # Paramètres
    SEQ_LENGTH = 10
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    HIDDEN_SIZE = 64
    features = ['log_moneyness', 'dte', 'hour_sin', 'hour_cos',
                'minutes_since_open', 'spot_price', 'right_enc']
    target = 'iv'

    # Connexion DB
    PG_URI = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(PG_URI)
    query = '''SELECT * FROM option_features WHERE iv IS NOT NULL ORDER BY ts_utc ASC;'''
    df = pd.read_sql(query, engine)
    df['right_enc'] = df['right'].map({'C': 0, 'P': 1}).astype(int)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Dataset
    class VolDataset(Dataset):
        def __init__(self):
            self.X, self.y = [], []
            grouped = df.groupby(['maturity', 'strike', 'right'])
            for _, group in grouped:
                group = group.sort_values('ts_utc')
                data = group[features + [target]].values
                for i in range(len(data) - SEQ_LENGTH):
                    self.X.append(data[i:i+SEQ_LENGTH, :-1])
                    self.y.append(data[i+SEQ_LENGTH, -1])
            self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
            self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    loader = DataLoader(VolDataset(), batch_size=BATCH_SIZE, shuffle=True)

    # Modèle LSTM
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze()

    model = LSTMModel(input_size=len(features), hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # MLflow
    experiment_name = "lstm_volatility_model"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params({
            "model": "LSTM",
            "seq_length": SEQ_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "lr": LR,
            "epochs": EPOCHS
        })

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
            avg_loss = total_loss / len(loader.dataset)
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch+1)

        mlflow.log_metric("final_loss", avg_loss)
        mlflow.pytorch.log_model(model, artifact_path="model")
