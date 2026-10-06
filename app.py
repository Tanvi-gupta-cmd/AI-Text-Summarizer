from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import re
from fastapi.responses import HTMLResponse

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

    api_url = "https://router.huggingface.co/hf-inference/models/t5-small"

    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": "summarize: " + dialogue
    }

    response = requests.post(
        api_url,
        headers=headers,
        json=payload,
        timeout=120
    )

    result = response.json()

    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "")

    if isinstance(result, dict) and "error" in result:
        return "Hugging Face error: " + result["error"]

    return "Unable to generate summary."


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