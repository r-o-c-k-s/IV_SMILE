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
import joblib

# ✅ Set MLflow tracking URI
uri = os.getenv("MLFLOW_TRACKING_URI", "file:/app/mlruns").replace("file:///", "file:/")
mlflow.set_tracking_uri(uri)
print("👉 Tracking URI:", mlflow.get_tracking_uri())

def train_gru():
    # 🔌 PostgreSQL connection
    PG_URI = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(PG_URI)

    # 🔧 Hyperparameters
    SEQ_LENGTH = 10
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    EPOCHS = 10
    LR = 0.001

    # 📥 Load and preprocess data
    query = '''
    SELECT * FROM option_features
    WHERE iv IS NOT NULL
    ORDER BY ts_utc ASC;
    '''
    df = pd.read_sql(query, engine)
    df['right_enc'] = df['right'].map({'C': 0, 'P': 1}).astype(int)

    features = ['log_moneyness', 'dte', 'hour_sin', 'hour_cos',
                'minutes_since_open', 'spot_price', 'right_enc']
    target = 'iv'

    # ✅ Standardize features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # 📆 Prepare sequences grouped by (maturity, strike, right)
    class VolDataset(Dataset):
        def __init__(self, df, seq_len):
            self.X, self.y = [], []
            grouped = df.groupby(['maturity', 'strike', 'right'])
            for _, group in grouped:
                group = group.sort_values('ts_utc')
                data = group[features + [target]].values
                for i in range(len(data) - seq_len):
                    seq_x = data[i:i+seq_len, :-1]
                    seq_y = data[i+seq_len, -1]
                    self.X.append(seq_x)
                    self.y.append(seq_y)
            self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
            self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    dataset = VolDataset(df, SEQ_LENGTH)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"📊 Total sequences: {len(dataset)}")

    # 🧠 GRU Model Definition
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, h = self.gru(x)
            out = self.fc(h[-1])
            return out.squeeze()  # ⚠️ ReLU removed for better flexibility

    model = GRUModel(input_size=len(features), hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 🚀 MLflow experiment setup
    experiment_name = "gru_volatility_model"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # 📦 Start MLflow run
    with mlflow.start_run():
        mlflow.log_params({
            "model": "GRU",
            "seq_length": SEQ_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS
        })

        # 🔁 Training loop
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch+1)

        mlflow.log_metric("final_loss", avg_loss)

        # ✅ Save scaler into model folder
        os.makedirs("model", exist_ok=True)
        scaler_path = os.path.join("model", "scaler.pkl")
        joblib.dump(scaler, scaler_path)

        # ✅ Log both model and scaler under 'model'
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_artifact(scaler_path, artifact_path="model")

if __name__ == "__main__":
    train_gru()
