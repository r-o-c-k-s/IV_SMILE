# Prédiction de la volatilité des options SPY

Projet académique réalisé dans le cadre d’un cours d’intelligence artificielle.  
Objectif : prédire la volatilité des options SPY à l’aide de modèles de machine learning.

## Contributions personnelles
- Définition du **business case** et cadrage du projet.  
- Participation à l’**analyse exploratoire** des données financières.  
- Vérification et **correction de code** (en pair programming, suggestions envoyées).  
- Documentation et préparation de la présentation.  

## Collaborateurs
Projet réalisé en collaboration avec [KAMOURI018](https://github.com/KAMOURI018).


# 📈 Pipeline de Prédiction du Volatility Smile du SPY en Temps Réel

Ce projet met en œuvre une architecture complète pour la prédiction du *volatility smile* du SPY en temps réel à l'aide de modèles d'IA (GRU, LSTM, MLP, Transformer). Il inclut la collecte des données via l’API TWS d’IBKR, le streaming temps réel avec Kafka, le stockage avec TimescaleDB, l’ingénierie de caractéristiques, l'entraînement de modèles, la prédiction temps réel et la visualisation dynamique avec Streamlit.

---

## 🔧 Architecture Globale Détaillée

```
            +---------------------+
            | API TWS (IBKR)      |  ← Données options SPY
            +----------+----------+
                       |
         +-------------v-------------+
         | Producteur Kafka (API)    |  ← Envoie vers Kafka topic
         +-------------+-------------+
                       |
         +-------------v-------------+
         | Broker Kafka (Docker)     |
         +-------------+-------------+
                       |
         +-------------v------------------------------+
         | Consommateurs Kafka (Dockerisés)           |
         | - Insertion dans TimescaleDB               |
         | - Feature Engineering                      |
         +----------------+---------------------------+
                          |
         +----------------v--------------------------+
         | - Prédicteur Temps Réel (GRU)              |
         +----------------+---------------------------+
                          |
              +-----------v------------+
              | TimescaleDB (Postgres) | ← Stockage structuré
              +-----------+------------+
                          |
              +-----------v-----------------+
              | Dashboard Streamlit         | ← Visualisation temps réel
              +-----------------------------+
```

---

## ⚙️ Composants

### 1. **Producteur Kafka**
Récupère la chaîne d’options SPY via l’API TWS d’Interactive Brokers toutes les 2 secondes et envoie les données à Kafka.

### 2. **Consommateurs Kafka**
- Écrivent les données dans `spy_option_chain`.
- Effectuent de l’ingénierie de caractéristiques et remplissent la table `option_features`.

### 3. **Ingénierie de Caractéristiques**
Transforme les données brutes en features exploitables pour l’entraînement et l’inférence (log-moneyness, DTE, encoding horaire, etc.).

### 4. **Entraîneur de Modèles**
Entraîne les modèles (GRU, LSTM, MLP, Transformer) et enregistre les modèles + scalers avec MLflow.

### 5. **Prédicteur Temps Réel**
Charge le modèle MLflow et le StandardScaler pour inférer l’IV en temps réel à partir de `option_features`. Résultats enregistrés dans `predicted_smile`.

### 6. **Dashboard Streamlit**
Affiche en direct les courbes de volatility smile à partir des prédictions stockées dans `predicted_smile` (via TimescaleDB).

---

## 🧠 Modèles d’IA

Plusieurs modèles peuvent être entraînés et testés :
- GRU (par défaut pour l’inférence temps réel)
- LSTM
- MLP
- Transformer

Les modèles sont suivis avec **MLflow**.

---

## 🗄️ Tables TimescaleDB

- `spy_option_chain` : données brutes en direct
- `option_features` : données transformées (features)
- `predicted_smile` : prédictions IV temps réel

---

## 📊 Streamlit Dashboard

```bash
cd dashboard
streamlit run app.py
```

Affiche en temps réel les smiles IV par maturité. Lecture directe depuis TimescaleDB.

---

## ⚙️ CI/CD avec GitHub Actions

Le projet inclut un pipeline CI/CD minimal basé sur GitHub Actions :

### `.github/workflows/main.yml`

- Checkout du repo
- Lint Python (`flake8`)
- Tests (`pytest` si présent)
- Build des containers Docker
- Optionnel : déploiement vers EC2 ou ECS avec secrets GitHub

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
          pip install -r predictor/requirements.txt

      - name: Run tests
        run: |
          pytest

      - name: Build Docker Images
        run: docker-compose build
```

---

## 🧪 Installation Locale

```bash
git clone https://github.com/KAMOURI018/IV_SMILE.git
cd IV_SMILE
docker-compose up --build
```

> **Remarque :** Assurez-vous que l’IB Gateway fonctionne localement.

---

## 📁 Structure du Projet

```
.
├── producer/                  # Producteur Kafka (API IBKR)
├── consumer/                  # Ingénierie de features + Insertion DB
├── predictor/                 # Inférence temps réel (modèle GRU)
├── dashboard/                 # Streamlit pour visualiser les smiles
├── mlruns/                    # Logs MLflow
├── docker-compose.yml
├── .github/                   # CI/CD workflows
├── README.md
```

---

## 📬 Contact

**Saad Rik**  
FRM | AI pour la Finance  
📍 Montréal, Canada  
✉️ riksaad@gmail.com  

