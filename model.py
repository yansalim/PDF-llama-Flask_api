from flask import Flask, request, jsonify
import os
import requests
import zipfile
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PyPDF2 import PdfReader

# Load environment variables from .env
load_dotenv()

# Configurations
PORT = int(os.getenv("PORT", 8080))
MODEL_DIR = os.getenv("MODEL_DIR", "./Llama-3.2-1B")
MODEL_URL = os.getenv("MODEL_URL", "https://example.com/model.zip")
MAX_BLOCK_SIZE = int(os.getenv("MAX_BLOCK_SIZE", 3000))
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

# AWS Credentials
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Setup S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION
)

# Function to upload file to S3
def upload_to_s3(file_path, file_name):
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, file_name)
        file_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{file_name}"
        return file_url
    except NoCredentialsError:
        raise ValueError("AWS credentials not configured or invalid.")

# Function to extract text from PDF URL and segment it
def extract_and_segment_text_from_url(pdf_url):
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        with open("temp.pdf", "wb") as temp_pdf:
            for chunk in response.iter_content(chunk_size=8192):
                temp_pdf.write(chunk)
        
        reader = PdfReader("temp.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        os.remove("temp.pdf")

        # Segment text into smaller blocks
        text_segments = [text[i:i + MAX_BLOCK_SIZE] for i in range(0, len(text), MAX_BLOCK_SIZE)]
        return text_segments
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")

# Ensure the model is available
def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory '{MODEL_DIR}' not found. Downloading...")
        zip_path = "llama_model.zip"
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete. Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
        except Exception as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise e

download_and_extract_model()

# Setup model and text generation pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map="auto")
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Flask App
app = Flask("Llama Server")

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        file_path = os.path.join('./', file.filename)
        file.save(file_path)

        # Upload to S3
        file_url = upload_to_s3(file_path, file.filename)

        # Remove local file
        os.remove(file_path)

        return jsonify({"message": "File uploaded successfully.", "url": file_url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    data = request.get_json()
    if 'url' not in data:
        return jsonify({"error": "Missing required parameter: 'url'"}), 400
    
    pdf_url = data['url']
    try:
        text_segments = extract_and_segment_text_from_url(pdf_url)
        return jsonify({"segments": len(text_segments), "message": "PDF processed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-text', methods=['POST'])
def generate_text_from_pdfs():
    data = request.get_json()
    if 'url' not in data or 'prompt' not in data or 'segment_index' not in data:
        return jsonify({"error": "Missing required parameters: 'url', 'prompt', and 'segment_index'"}), 400
    
    pdf_url = data['url']
    segment_index = int(data['segment_index'])
    prompt = data['prompt']
    
    try:
        text_segments = extract_and_segment_text_from_url(pdf_url)

        if segment_index < 0 or segment_index >= len(text_segments):
            return jsonify({"error": f"Segment index out of range. Total segments: {len(text_segments)}"}), 400

        text_segment = text_segments[segment_index]
        combined_prompt = f"{prompt}\n\n{text_segment}"

        response = text_gen(
            combined_prompt,
            max_length=200,
            num_return_sequences=1,
            do_sample=True,
            top_k=10,
            eos_token_id=tokenizer.eos_token_id
        )
        return jsonify({"generated_text": response[0]['generated_text']}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)