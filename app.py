```python
from fastapi import FastAPI
from pydantic import BaseModel
import os
import re
from fastapi.responses import HTMLResponse
from huggingface_hub import InferenceClient

# initialize fastapi app
app = FastAPI(
    title="Text Summarizer App",
    description="Text summarization using T5 deployed by Ishwari Daphal",
    version="1.0"
)

# input schema
class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    client = InferenceClient(
        provider="hf-inference",
        api_key=os.getenv("HF_TOKEN")
    )

    result = client.summarization(
        dialogue,
        model="facebook/bart-large-cnn"
    )

    return result.summary_text


# API endpoint
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


# Home page
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()
```
