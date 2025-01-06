# Use python as base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the application files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gcc \
    g++ \
    procps \
    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers \
    Flask \
    llama-cpp-python \
    torch \
    tensorflow \
    flax \
    sentencepiece \
    huggingface_hub \
    accelerate \
    PyPDF2 \
    boto3 \
    python-dotenv

# Copy .env file to the container
COPY .env /app/.env

# Expose the port used by the Flask app
EXPOSE 8080

# Set the default command
CMD ["python", "model.py"]