import os
import zipfile
import gdown
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# -----------------------------
# Serve Frontend
# -----------------------------
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

# -----------------------------
# Model Download Configuration
# -----------------------------
MODEL_DIR = "data"
ZIP_FILE = "data.zip"
FILE_ID = "1KJzGif2a1HqSWGwOoqNMTWCS_wz65Ba0"
MODEL_FILE = os.path.join(MODEL_DIR, "model.safetensors")

def download_and_extract_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model zip from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        gdown.download(url, ZIP_FILE, quiet=False)

        print("Extracting model...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".")

        print("Model extracted successfully!")

# -----------------------------
# Model Setup (Lazy Loading)
# -----------------------------
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        # Ensure model files exist
        download_and_extract_model()

        print("Loading model into memory...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        model.to(device)
        print("Model loaded successfully!")

# -----------------------------
# Request Schema
# -----------------------------
class TranslationRequest(BaseModel):
    text: str
    language: str   # hindi or punjabi

# -----------------------------
# Translation API
# -----------------------------
@app.post("/translate")
def translate(request: TranslationRequest):

    load_model()  # loads model only first time

    if request.language == "hindi":
        prefix = "translate English to Hindi: "
    elif request.language == "punjabi":
        prefix = "translate English to Punjabi: "
    else:
        prefix = ""

    input_text = prefix + request.text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=100,
        num_beams=4,
        repetition_penalty=2.5
    )

    translated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {"translation": translated_text}

