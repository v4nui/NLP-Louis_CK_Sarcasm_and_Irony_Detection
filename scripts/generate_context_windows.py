import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download required tokenizer data (if not already downloaded)
nltk.download('punkt')

# Load cleaned sentences
df = pd.read_csv('data/louis_ck_clean_sentences.csv')

# Define window size
window_size = 12

context_windows = []
video_ids = df['video_id'].unique()

for vid in video_ids:
    sentences = df[df['video_id'] == vid]['sentence'].tolist()
    
    for i in range(len(sentences) - window_size + 1):
        window = sentences[i:i+window_size]
        context_text = ' '.join(window)
        context_windows.append({
            'video_id': vid,
            'start_sentence_idx': i,
            'context_window': context_text
        })

# Convert context windows into a DataFrame
df_context = pd.DataFrame(context_windows)

# Tokenize the context window texts
df_context['tokens'] = df_context['context_window'].apply(word_tokenize)

# Save for annotation
df_context.to_csv('data/louis_ck_context_windows_tok.csv', index=False)

print("Context windows with tokenization saved successfully!")
