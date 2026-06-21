import os
import pandas as pd

def run_task1_streaming(raw_path, output_path):
    print("🚀 Task 1: Starting Scalable Stream Preprocessing Pipeline...")
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Please place your complaints.csv there.")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Category mapping schema
    category_map = {
        'Credit card or prepaid card': 'Credit Card', 'Credit card': 'Credit Card',
        'Checking or savings account': 'Savings Account', 'Savings account': 'Savings Account',
        'Consumer Loan': 'Personal Loan', 'Vehicle loan or lease': 'Personal Loan',
        'Money transfer, virtual currency, or money service': 'Money Transfer'
    }
    
    first_chunk = True
    # Stream in 50k blocks to preserve low system RAM footprints
    for chunk in pd.read_csv(raw_path, chunksize=50000, low_memory=False):
        # Filter and map
        chunk_filtered = chunk[chunk['Product'].isin(category_map.keys())].copy()
        chunk_filtered['product_category'] = chunk_filtered['Product'].map(category_map)
        
        # Clean naming consistency
        chunk_filtered = chunk_filtered.rename(columns={
            'Complaint ID': 'complaint_id',
            'Product': 'product',
            'Issue': 'issue',
            'Sub-issue': 'sub_issue',
            'Company': 'company',
            'Consumer complaint narrative': 'cleaned_narrative'
        })
        
        # Isolate core attributes and clean narrative strings
        columns_to_keep = ['complaint_id', 'product_category', 'product', 'issue', 'sub_issue', 'company', 'cleaned_narrative']
        chunk_filtered = chunk_filtered[columns_to_keep].dropna(subset=['cleaned_narrative'])
        
        # Normalize text structural tokens and scrub PII tags
        chunk_filtered['cleaned_narrative'] = chunk_filtered['cleaned_narrative'].astype(str).str.lower()
        chunk_filtered['cleaned_narrative'] = chunk_filtered['cleaned_narrative'].str.replace(r'x+/x+/x+', '', regex=True).str.replace(r'x+', '', regex=True)
        chunk_filtered = chunk_filtered[chunk_filtered['cleaned_narrative'].str.split().str.len() > 3]
        
        # Append to target storage file dynamically
        if first_chunk:
            chunk_filtered.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk_filtered.to_csv(output_path, index=False, mode='a', header=False)
            
    print(f"✅ Success! Filtered stream compiled nicely to: {output_path}")

if __name__ == "__main__":
    run_task1_streaming("data/raw/complaints.csv", "data/processed/filtered_complaints.csv")