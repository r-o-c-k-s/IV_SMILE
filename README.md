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