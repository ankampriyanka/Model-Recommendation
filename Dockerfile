FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    cmake \
    libomp-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install python packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /app/requirements.txt

# Copy application files
COPY . /app

# Expose streamlit port
EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
