from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import os

video_ids = [
    'TBg_yJ1TJRE',
    '_FEQvVoiarg',
    'qALqXrAS9Ug'
]

sentences = []

for vid in video_ids:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(vid, languages=['en'])
        
        for item in transcript_list:
            text = item['text'].replace('\n', ' ').strip()
            if text:  # Ignore empty lines
                sentences.append({'video_id': vid, 'sentence': text})

        print(f"Successfully extracted and split sentences for video ID: {vid}")

    except Exception as e:
        print(f"Error extracting transcript for video ID {vid}: {e}")

# Convert to DataFrame
df_sentences = pd.DataFrame(sentences)

# Save directly into existing 'data' folder
output_dir = './data'
os.makedirs(output_dir, exist_ok=True)

# Save extracted sentences directly
df_sentences.to_csv(os.path.join(output_dir, 'louis_ck_sentences.csv'), index=False)
print("Transcript sentences saved successfully!")
