import pandas as pd
import re

# Load sentence-level transcript file
df = pd.read_csv('data/louis_ck_sentences.csv')

# Define tags to remove
TAGS_TO_REMOVE = [r'\[Applause\]', r'\[Music\]']

# Function to clean tags and replace censored placeholders
def clean_text(text):
    # Remove noise tags like [Applause]
    for tag in TAGS_TO_REMOVE:
        text = re.sub(tag, '', text, flags=re.IGNORECASE)

    # Replace [ __ ] or similar with [CENSORED]
    text = re.sub(r'\[\s*__\s*\]', '[CENSORED]', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply cleaning
df['sentence'] = df['sentence'].apply(clean_text)

# Drop rows that are now empty (either NaN or empty strings)
df.dropna(subset=['sentence'], inplace=True)
df = df[df['sentence'].str.strip() != '']

# Save the cleaned version
df.to_csv('data/louis_ck_clean_sentences.csv', index=False)

print("Cleaned file saved as louis_ck_clean_sentences.csv")
