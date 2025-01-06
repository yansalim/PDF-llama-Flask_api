# 🦙 Llama API - PDF Processing and Text Generation

This project provides an API to upload PDFs to an Amazon S3 bucket, process them by extracting and segmenting text, and generate responses using Meta's Llama language model. 

---

## 🚀 Features

- Upload PDF files to an S3 bucket.
- Process and segment large PDFs into manageable chunks.
- Generate responses using Llama's text-generation capabilities.
- Environment variables for secure configuration.

---

## 🛠️ Prerequisites

- **Python 3.9+**
- **Docker** and **Docker Compose** installed.
- An AWS S3 bucket and access credentials.
- The Llama model downloaded or available for automatic download.

---

## 🐍 Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/llama-api.git
cd llama-api

2. Create the .env File

use example.env

3. pip install -r requirements.txt

python model.py

🐳 Running with Docker

1. Build the Docker Image
docker build -t llama-api .

2. Run the Docker Container
docker run -p 8080:8080 --env-file .env llama-api