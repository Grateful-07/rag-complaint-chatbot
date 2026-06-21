import os
import re
import pandas as pd

def clean_narrative(text):
    """
    Cleans raw CFPB complaint narratives by lowercasing, 
    removing structural privacy redactions, and stripping boilerplate.
    """
    if pd.isna(text):
        return ""
    
    text = text.lower()
    
    # Strip CFPB privacy masks (e.g., XXXX, XX/XX/XXXX)
    text = re.sub(r'x{2,}', '', text)
    text = re.sub(r'\d{2}/\d{2}/\d{4}', '', text)
    
    # Strip annoying standard introductory boilerplate phrases
    boilerplate_patterns = [
        r"i am writing to file a complaint regarding",
        r"to whom it may concern",
        r"i am a victim of"
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)
        
    # Clean up special characters, punctuation, and extra whitespace
    text = re.sub(r'[^a-z0-9\s.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_task1_pipeline(input_path, output_path):
    print("🚀 Task 1: Starting memory-efficient Streaming Preprocessing...")
    
    # Map raw CFPB categories directly to CrediTrust internal business products
    product_mapping = {
        'Credit card or prepaid card': 'Credit Card',
        'Credit card': 'Credit Card',
        'Checking or savings account': 'Savings Account',
        'Savings account': 'Savings Account',
        'Consumer Loan': 'Personal Loan',
        'Vehicle loan or lease': 'Personal Loan',
        'Money transfer, virtual currency, or money service': 'Money Transfer',
        'Money transfers': 'Money Transfer'
    }
    
    # Only pull these columns into memory to prevent Out-Of-Memory crashes
    required_cols = [
        'Complaint ID', 'Product', 'Sub-product', 'Issue', 
        'Sub-issue', 'Consumer complaint narrative', 'Company', 
        'State', 'Date received'
    ]
    
    chunk_size = 50000
    first_chunk = True
    total_reviewed = 0
    total_retained = 0
    
    # Initialize basic online aggregation counters for the EDA summary metrics
    product_counts = {}
    total_complaints_with_narratives = 0
    total_complaints_without_narratives = 0

    for chunk in pd.read_csv(input_path, usecols=required_cols, chunksize=chunk_size, low_memory=False):
        total_reviewed += len(chunk)
        
        # Calculate narrative presence metrics for EDA
        with_narrative = chunk['Consumer complaint narrative'].notna().sum()
        total_complaints_with_narratives += with_narrative
        total_complaints_without_narratives += (len(chunk) - with_narrative)
        
        # Drop missing narratives right away
        chunk = chunk.dropna(subset=['Consumer complaint narrative'])
        
        # Map to CrediTrust product categories
        chunk['product_category'] = chunk['Product'].map(product_mapping)
        chunk = chunk.dropna(subset=['product_category'])
        
        if chunk.empty:
            continue
            
        # Clean text
        chunk['cleaned_narrative'] = chunk['Consumer complaint narrative'].apply(clean_narrative)
        chunk['narrative_word_count'] = chunk['cleaned_narrative'].apply(lambda x: len(x.split()))
        
        # Remove low-value, empty narratives (under 4 words)
        chunk = chunk[chunk['narrative_word_count'] > 3]
        
        # Standardize schema names for the downstream RAG architecture
        chunk = chunk.rename(columns={
            'Complaint ID': 'complaint_id',
            'Sub-product': 'product',
            'Issue': 'issue',
            'Sub-issue': 'sub_issue',
            'Company': 'company',
            'State': 'state',
            'Date received': 'date_received'
        })
        
        # Keep track of category distributions for our EDA reporting
        for prod, count in chunk['product_category'].value_counts().items():
            product_counts[prod] = product_counts.get(prod, 0) + count
            
        final_cols = ['complaint_id', 'product_category', 'product', 'issue', 
                      'sub_issue', 'company', 'state', 'date_received', 'cleaned_narrative', 'narrative_word_count']
        output_chunk = chunk[final_cols]
        
        # Stream directly to output file
        if first_chunk:
            output_chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            output_chunk.to_csv(output_path, index=False, mode='a', header=False)
            
        total_retained += len(output_chunk)
        print(f"   Processed {total_reviewed} raw rows... Saved {total_retained} target records.")

    print("\n📊 --- Task 1 EDA Quick Metrics Summary ---")
    print(f"Total Raw Rows Evaluated: {total_reviewed}")
    print(f"Complaints WITH Narratives: {total_complaints_with_narratives}")
    print(f"Complaints WITHOUT Narratives: {total_complaints_without_narratives}")
    print(f"Final Retained Target Category Distribution: {product_counts}")
    print(f"✅ Success! File saved to {output_path}\n")


RAW_PATH = "data/raw/complaints.csv"
PROCESSED_PATH = "data/processed/filtered_complaints.csv"

run_task1_pipeline(RAW_PATH, PROCESSED_PATH)