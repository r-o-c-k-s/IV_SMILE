FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of your training scripts (GRU, MLP, etc.)
COPY . .

# Default command: train GRU model (can be overridden with --model MLP)
CMD ["python", "train_model.py", "--model", "GRU"]
