# -------------------- BASE IMAGE --------------------
FROM python:3.10-slim

# -------------------- ENV SETTINGS --------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------- WORKDIR --------------------
WORKDIR /app

# -------------------- SYSTEM DEPENDENCIES --------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------- PYTHON DEPENDENCIES --------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------- COPY PROJECT --------------------
COPY . .

# -------------------- EXPOSE PORT --------------------
EXPOSE 8080

# -------------------- RUN APP --------------------
CMD ["python", "app.py"]
