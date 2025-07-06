FROM python:3.10

WORKDIR /app

# ✅ Copy only requirements.txt first (for better caching)
COPY requirements.txt .

# ✅ Install both psycopg2 and all dependencies in one layer
RUN pip install --no-cache-dir psycopg2-binary && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Now copy the rest of the application code
COPY . .

# ✅ Run the predictor script
CMD ["python", "-u", "predict_smile_realtime.py"]
