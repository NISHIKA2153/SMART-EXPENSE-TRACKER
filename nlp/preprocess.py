import re

# optional: list of common names (for In Person noise)
names = [
    "rahul","neha","amit","priya","rohit",
    "anjali","saurabh","kavya","arjun","sneha",
    "vikas","pooja","deepak","riya"
]

def clean_text(text):
    text = str(text).lower()  # ensure string + lowercase

    # remove rs / inr
    text = re.sub(r'rs\.?|inr', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # remove names (important for better classification)
    for name in names:
        text = text.replace(name, '')

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
