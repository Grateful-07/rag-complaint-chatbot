import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def run_task2_pipeline(processed_csv_path, vector_store_dir):
    print("🚀 Task 2: Starting Proportional Stratified Sampling & FAISS Storing...")
    
    if not os.path.exists(processed_csv_path):
        raise FileNotFoundError(f"Processed file missing at {processed_csv_path}.")
        
    df = pd.read_csv(processed_csv_path)
    
    # Extract representative sample across target proportions
    target_sample_size = min(15000, len(df))
    df_sample, _ = train_test_split(df, train_size=target_sample_size, stratify=df['product_category'], random_state=42)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_text, chunks_metadata = [], []
    
    for _, row in df_sample.iterrows():
        individual_chunks = text_splitter.split_text(str(row['cleaned_narrative']))
        for text_chunk in individual_chunks:
            chunks_text.append(text_chunk)
            chunks_metadata.append({
                "complaint_id": row['complaint_id'], "product_category": row['product_category'],
                "product": row['product'], "issue": row['issue'], "company": row['company'], "text_chunk": text_chunk
            })
            
    print(f"   Generated {len(chunks_text)} semantic text chunks. Mapping embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.array(embedding_model.encode(chunks_text, show_progress_bar=True, batch_size=64)).astype('float32')
    
    # Persist directly into vector storage definitions
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    os.makedirs(vector_store_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(vector_store_dir, "complaints_faiss.index"))
    with open(os.path.join(vector_store_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(chunks_metadata, f)
        
    print(f"✅ Success! Local FAISS vector index saved to '{vector_store_dir}/'")

if __name__ == "__main__":
    run_task2_pipeline("data/processed/filtered_complaints.csv", "vector_store")