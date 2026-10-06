from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-small"

AutoTokenizer.from_pretrained(model_name)
AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model downloaded successfully!")