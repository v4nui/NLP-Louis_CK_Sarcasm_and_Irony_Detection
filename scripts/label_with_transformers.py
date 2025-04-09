import pandas as pd
from transformers import pipeline

# Load context windows training data
df = pd.read_csv('data\louis_ck_test.csv')

# Load pretrained sarcasm and irony detection pipeline
sarcasm_irony_detector = pipeline(
    'text-classification', 
    model='cardiffnlp/twitter-roberta-base-irony',
    tokenizer='cardiffnlp/twitter-roberta-base-irony'
)

# Load pretrained sentiment analysis pipeline
sentiment_analyzer = pipeline(
    'sentiment-analysis', 
    model='cardiffnlp/twitter-roberta-base-sentiment',
    tokenizer='cardiffnlp/twitter-roberta-base-sentiment'
)

# Apply sarcasm/irony detection
def detect_sarcasm_irony(text):
    result = sarcasm_irony_detector(text, truncation=True)[0]
    label = result['label']
    score = result['score']
    return pd.Series([label, score])

df[['irony_sarcasm_label', 'irony_sarcasm_score']] = df['context_window'].apply(detect_sarcasm_irony)

# Apply sentiment analysis
def detect_sentiment(text):
    result = sentiment_analyzer(text, truncation=True)[0]
    return pd.Series([result['label'], result['score']])

df[['sentiment_label', 'sentiment_score']] = df['context_window'].apply(detect_sentiment)

# Save auto-labeled file
df.to_csv('data/louis_ck_test_auto_labeled.csv', index=False)
print("Auto-labeling completed!")
