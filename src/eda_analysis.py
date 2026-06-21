import pandas as pd

# Load your clean, processed data
df = pd.read_csv("data/processed/filtered_complaints.csv")

print("=== 1. COMPLAINT DISTRIBUTION PER TARGET CATEGORY ===")
print(df['product_category'].value_counts())
print("\n=== PERCENTAGE PROPORTIONS ===")
print(df['product_category'].value_counts(normalize=True) * 100)

print("\n=== 2. NARRATIVE WORD COUNT METRICS ===")
word_counts = df['cleaned_narrative'].astype(str).str.split().str.len()
print(word_counts.describe())