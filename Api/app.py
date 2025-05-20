from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -------- Load Tokenizer & Model --------
MODEL_NAME = "csebuetnlp/banglabert"
NUM_LABELS = 5  # For 5-class classification

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Load fine-tuned weights
model.load_state_dict(torch.load("sentiment.bin", map_location=torch.device("cpu")))
model.eval()

# -------- FastAPI App Setup --------
app = FastAPI()

# Label map for human-readable output
label_map = {
    0: "sexual",
    1: "not bully",
    2: "troll",
    3: "religious",
    4: "threat"
}

# Request schema
class TextInput(BaseModel):
    text: str

# -------- Prediction Endpoint --------
@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

    return {
        "prediction": label_map[pred_label],
        "confidence": round(probs[0][pred_label].item(), 4),
        "all_probabilities": {label_map[i]: round(p, 4) for i, p in enumerate(probs[0].tolist())}
    }