# IV_SMILE 📈 – Pipeline de Prédiction du Volatility Smile

Ce projet met en place un pipeline d'intelligence artificielle en temps réel pour prédire le **volatility smile** des options SPY à partir de données diffusées par Interactive Brokers (TWS), traitées via Kafka, et modélisées à l'aide de réseaux de neurones profonds (GRU, LSTM, MLP, Transformer).  
Les entraînements sont suivis via **MLflow** et les résultats sont stockés dans **TimescaleDB**.

---

## 🚀 Vue d'ensemble de l'architecture

```
TWS (API IB)
    ↓
Producteur Kafka (Python)
    ↓
Topic Kafka
    ↓
Consommateur Kafka + Feature Engineering
    ↓
TimescaleDB (table option_features)
    ↓
Modèles IA (GRU, LSTM, MLP, Transformer)
    ↓
MLflow (suivi des modèles, métriques, artefacts)
```

---

## 📦 Structure du projet

```
IV_SMILE/
├── producer/              # De TWS vers Kafka
├── consumer/              # De Kafka vers TimescaleDB (feature engineering)
├── model/                 # Modèles IA et logique d'entraînement
│   ├── gru_model.py
│   ├── lstm_model.py
│   ├── mlp_model.py
│   ├── transformer_model.py
│   ├── train_model.py
├── mlruns/                # Logs MLflow (exclu du Git)
├── docker-compose.yml     # Orchestration complète
├── explore_db.py          # Script pour explorer la base Timescale
├── README.md
└── .gitignore
```

---

## 🧠 Modèles utilisés

Les modèles suivants sont entraînés sur des séquences temporelles d’options SPY pour prédire la **volatilité implicite** :
- ✅ GRU
- ✅ LSTM
- ✅ MLP
- ✅ Transformer

Chaque modèle enregistre :
- Les hyperparamètres : `lr`, `seq_length`, `epochs`, etc.
- Les métriques : `epoch_loss`, `final_loss`
- Les artefacts : modèle `.pth`, fichiers de configuration MLflow

---

## 🔧 Conception des Features

### ✅ 1. `log_moneyness`
```python
log_moneyness = log(strike / spot_price)
```
- Capture la position relative du strike par rapport au sous-jacent.
- Utilisé dans la plupart des modèles de surface de volatilité.

---

### ✅ 2. `dte` – Jours avant expiration
```python
dte = (maturity_date - ts_utc).total_seconds() / (60 * 60 * 24)
```
- Temps restant avant l’expiration, en jours.
- Très important pour modéliser la dépréciation temporelle.

---

### ✅ 3. `right_enc`
```python
right_enc = 0 si call, 1 si put
```
- Encodage binaire du type d’option.

---

### ✅ 4. `hour_sin`, `hour_cos`
```python
Heure en sinus/cosinus sur 24h
```
- Permet de capturer les effets saisonniers intra-journaliers.

---

### ✅ 5. `minutes_since_open`
```python
minutes depuis l'ouverture à 9h30
```
- Capture la dynamique d’ouverture/fermeture des marchés.

---

### 🧪 Vecteur de features final
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
- Normalisé avec `StandardScaler`
- Groupé par `(maturity, strike, right)`
- Utilisé sur des séquences temporelles de longueur `SEQ_LENGTH`

---

### 🎯 Cible
```python
target = 'iv'
```
On cherche à prédire la volatilité implicite au pas de temps suivant.

---

## 🐳 Lancer le projet

### Entraîner tous les modèles avec Docker
```bash
for model in GRU LSTM MLP TRANSFORMER; do
  docker-compose run --rm model-trainer python train_model.py --model $model
done
```

### Lancer l'interface MLflow
```bash
mlflow ui --backend-store-uri ./mlruns
```

➡ Puis ouvre : http://127.0.0.1:5000/

---

## 💾 Interface MLflow
Tu peux y comparer les modèles, consulter les courbes de perte, et télécharger les modèles `.pth`.

---

## ✅ Améliorations futures
- [ ] `predict_smile.py` — Inférence avec modèles MLflow
- [ ] API FastAPI ou Flask pour servir les modèles
- [ ] Visualisation du smile en temps réel
- [ ] Déploiement cloud (AWS, GCP)
- [ ] Démo interactive en notebook

---

## 🧪 Dépendances
Chaque composant contient un `requirements.txt`. Tu peux les installer manuellement ou tout lancer via Docker.

---

## 📜 Licence
MIT © Khalil Amouri — Contributions bienvenues.