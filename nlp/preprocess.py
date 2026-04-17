import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'rs\.?\s?\d+', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.strip()