# IV_SMILE 📈 – Volatility Smile Prediction Pipeline

This project builds a real-time machine learning pipeline to predict the **volatility smile** of SPY options using data streamed from Interactive Brokers (TWS), processed via Kafka, and modeled with deep learning architectures (GRU, LSTM, MLP, Transformer).  
Training runs are tracked using **MLflow** and results are stored in **TimescaleDB**.

---

## 🚀 Architecture Overview

```
TWS (IB API)
    ↓
Kafka Producer (Python)
    ↓
Kafka Topic
    ↓
Kafka Consumer + Feature Engineering
    ↓
TimescaleDB (option_features table)
    ↓
Deep Learning Models (GRU, LSTM, MLP, Transformer)
    ↓
MLflow (model tracking, metrics, artifacts)
```

---

## 📦 Project Structure

```
IV_SMILE/
├── producer/            # TWS to Kafka
├── consumer/            # Kafka to TimescaleDB (feature engineering)
├── model/               # Deep learning models and training logic
│   ├── gru_model.py
│   ├── lstm_model.py
│   ├── mlp_model.py
│   ├── transformer_model.py
│   ├── train_model.py
├── mlruns/              # MLflow run logs (excluded from Git)
├── docker-compose.yml   # Full pipeline orchestration
├── explore_db.py        # Script to explore the DB manually
├── README.md
└── .gitignore
```

---

## 🧠 Models


The following models are trained on time-sequenced SPY option data to predict **implied volatility**:
- ✅ GRU
- ✅ LSTM
- ✅ MLP
- ✅ Transformer

Each model logs:
- Parameters: `lr`, `seq_length`, `epochs`, etc.
- Metrics: `epoch_loss`, `final_loss`
- Artifacts: saved `.pth` models, `MLmodel` metadata, environments

---

## 🐳 Running the Project

### Train all models with Docker
```bash
for model in GRU LSTM MLP TRANSFORMER; do
  docker-compose run --rm model-trainer python train_model.py --model $model
done
```

### Launch MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns
```
➡ Then open: http://127.0.0.1:5000/

---

## 💾 MLflow UI
You can compare runs, view metrics and download models from the MLflow interface.

---

## ✅ TODO (Next Enhancements)
- [ ] `predict_smile.py` — Inference from saved models
- [ ] FastAPI or Flask service to serve models in real-time
- [ ] Add visualization for volatility smile curves
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Integrate notebook-based demos for presentations

---

## 🧪 Requirements
All dependencies are inside each component’s `requirements.txt`. You can install them or use Docker for isolation.

---

## 📜 License
MIT © Khalil Amouri — Feel free to contribute or fork.

## 🧠 Feature Engineering Design

The goal of feature engineering in this project is to help sequential models (GRU, LSTM, Transformer, etc.) learn the dynamic behavior of the **volatility smile** by combining market structure information with temporal context.

---

### ✅ 1. `log_moneyness`
```python
log_moneyness = log(strike / spot_price)
```
- Captures the relative position of the strike to the underlying.
- Commonly used in volatility surface modeling.

---

### ✅ 2. `dte` – Days to Expiration
```python
dte = (maturity_date - ts_utc).total_seconds() / (60 * 60 * 24)
```
- Measures the remaining time until option expiration in days.
- Crucial for modeling time decay effects.

---

### ✅ 3. `right_enc`
```python
right_enc = 0 if right == 'C' else 1
```
- Binary encoding for option type: 0 = Call, 1 = Put.
- Enables model to distinguish behaviors of puts vs calls.

---

### ✅ 4. `hour_sin`, `hour_cos`
```python
hour = ts_utc.hour + ts_utc.minute / 60
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
```
- Cyclical encoding of time-of-day (e.g., 9:30 AM vs 3:00 PM).
- Helps model intraday seasonality in volatility.

---

### ✅ 5. `minutes_since_open`
```python
minutes_since_open = (ts_utc.hour - 9) * 60 + ts_utc.minute - 30
```
- Measures how long since market open (9:30 AM).
- Captures volatility clustering around open and close.

---

### 🧪 Final Feature Set

```python
features = [
    'log_moneyness',
    'dte',
    'hour_sin',
    'hour_cos',
    'minutes_since_open',
    'spot_price',
    'right_enc'
]
```

- Features are scaled with `StandardScaler`.
- Grouped by `(maturity, strike, right)` to form temporal sequences.
- Used to build time-series inputs of length `SEQ_LENGTH`.

---

### 🎯 Target Variable

```python
target = 'iv'
```
The model predicts the implied volatility of the option **at the next time step**.