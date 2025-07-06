# 📈 Real-Time SPY Volatility Smile Prediction Pipeline

This project is a full pipeline for real-time prediction of the SPY volatility smile using AI models (GRU, LSTM, MLP, Transformer). It includes ingestion of options market data from IBKR's TWS API, real-time streaming via Kafka, storage in TimescaleDB, feature engineering, ML training, and visualization via Streamlit.

---

## 🔧 Architecture Overview

```
            +--------------------+
            |  TWS API (IBKR)   |
            +--------+-----------+
                     |
         +-----------v------------+
         |  Kafka Producer (API)  |
         +-----------+------------+
                     |
         +-----------v------------+
         | Kafka Broker (Docker)  |
         +-----------+------------+
                     |
         +-----------v--------------------+
         | Kafka Consumers (Dockerized)  |
         |  - Data Writer to TimescaleDB |
         |  - Feature Engineering        |
         |  - Real-time Predictor (GRU)  |
         +-------------------------------+
                     |
         +-----------v-------------+
         | TimescaleDB (Postgres) |
         +-----------+-------------+
                     |
         +-----------v----------------+
         | Streamlit Dashboard (Live) |
         +----------------------------+
```

---

## 📦 Components

### 1. **Kafka Producer**
Fetches SPY option chain data from IB TWS API and sends messages to a Kafka topic every 2 seconds.

### 2. **Kafka Consumer**
Consumes option chain messages and inserts data into TimescaleDB (`spy_option_chain`).

### 3. **Feature Engineering**
Processes `spy_option_chain` to generate engineered features in `option_features` table.

### 4. **Model Trainer**
Trains AI models (GRU, etc.) on historical data and logs with MLflow.

### 5. **Real-Time Predictor**
Loads trained model and StandardScaler from MLflow and performs live inference on `option_features`, writing results to `predicted_smile`.

### 6. **Streamlit Dashboard**
Visualizes the volatility smile curve in real time using data from `predicted_smile`.

---

## 🧠 Models

You can choose and train:
- GRU (default for real-time)
- LSTM
- MLP
- Transformer

The model is saved in MLflow and used in production by the predictor service.

---

## 📊 TimescaleDB Tables

- `spy_option_chain`: Raw option chain data
- `option_features`: Engineered features
- `predicted_smile`: Real-time IV predictions

---

## 🚀 CI/CD (GitHub Actions)

CI/CD is managed using GitHub Actions:

### Workflow: `.github/workflows/main.yml`

It includes:
- Docker image build for all services
- Python linting with `flake8`
- Unit tests (if `tests/` folder exists)
- Deployment steps (manual or via GitHub runners)

### Example snippet:

```yaml
name: Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r predictor/requirements.txt

      - name: Run tests
        run: |
          pytest

      - name: Build Docker Images
        run: docker-compose build
```

> You can customize deployment to EC2, ECS, or any cloud provider using secrets and runners.

---

## 📦 Installation

```bash
git clone https://github.com/youruser/volatility-pipeline.git
cd volatility-pipeline
docker-compose up --build
```

Make sure IB Gateway is running locally and properly connected.

---

## 📈 Streamlit Dashboard

```bash
cd dashboard
streamlit run app.py
```

The dashboard fetches `predicted_smile` data from TimescaleDB and updates live.

---

## 📁 Directory Structure

```
.
├── producer/                  # IBKR Kafka producer
├── consumer/                  # Kafka consumer & feature engineering
├── predictor/                 # Real-time model inference
├── dashboard/                 # Streamlit app
├── mlruns/                    # MLflow experiments
├── docker-compose.yml
├── README.md
```

---

## 📬 Contact

Khalil Amouri  
FRM | AI for Finance  
📍 Montréal, Canada  
✉️ cashcouscous.ai@gmail.com  
TikTok: [@cashcouscous](https://tiktok.com/@cashcouscous)