from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize fastapi app
app = FastAPI(title="Text Summarizer App", description="Text summarization using T5 deployed by Ishwari Daphal", version="1.0")

# model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory="templates")

# input schema
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue)
    input_text = "summarize: " + dialogue 

    #tokenize
    inputs = tokenizer(
        input_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    # generate summary => token ids
    targets = model.generate(
        inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # token ids to text
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)

    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/")
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/")
def test():
    return {"msg": "server working"}