import pandas as pd
from sklearn.model_selection import train_test_split

# Load windowed data set
df = pd.read_csv('data/louis_ck_context_windows_tok.csv')

# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)

# Second split: split the 30% into 15% val, 15% test
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

# Save the splits
train_df.to_csv('data/louis_ck_train.csv', index=False)
val_df.to_csv('data/louis_ck_val.csv', index=False)
test_df.to_csv('data/louis_ck_test.csv', index=False)

print(f"Dataset split completed:")
print(f"Training set: {len(train_df)} rows")
print(f"Validation set: {len(val_df)} rows")
print(f"Test set: {len(test_df)} rows")