import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/filtered_complaints.csv")

# Chart 1: Complaint Distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='product_category', order=df['product_category'].value_counts().index, palette='viridis')
plt.title('Proportional Distribution of Consumer Financial Complaints')
plt.xlabel('Number of Complaints')
plt.ylabel('Product Category')
plt.tight_layout()
plt.savefig('complaint_distribution.png')
plt.close()

# Chart 2: Narrative Word Count Distribution
word_counts = df['cleaned_narrative'].astype(str).str.split().str.len()
plt.figure(figsize=(10, 5))
sns.histplot(word_counts, bins=50, kde=True, color='blue')
plt.xlim(0, 500)  # Limit to 500 words to clearly show the main distribution
plt.title('Distribution of Cleaned Complaint Narrative Word Counts')
plt.xlabel('Word Count per Narrative')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('narrative_word_counts.png')
plt.close()
print("🎉 Charts saved successfully as 'complaint_distribution.png' and 'narrative_word_counts.png'!")