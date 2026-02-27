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
# Model Setup (Lazy Loading)
# -----------------------------
MODEL_PATH = "data"

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading model...")
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

    load_model()  # load only first time

    # Select prefix based on language
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