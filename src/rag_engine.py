import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate

class CrediTrustRAG:
    def __init__(self, parquet_path="data/complaint_embeddings.parquet"):
        print("🧠 Loading pre-built full-scale embeddings from Parquet...")
        self.parquet_path = parquet_path
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found at {parquet_path}. Please verify the file location.")
        
        # 1. Load the full pre-computed embeddings dataset
        self.df = pd.read_parquet(parquet_path)
        
        # 2. Extract the embeddings vector column into an easily searchable numpy matrix
        # (Assuming the vector column name is 'embeddings' or similar; adjust if named differently)
        embedding_col = 'embeddings' if 'embeddings' in self.df.columns else [col for col in self.df.columns if 'embed' in col.lower()][0]
        
        print("⚡ Compiling vector lookup arrays...")
        self.embeddings_matrix = np.vstack(self.df[embedding_col].values).astype('float32')
        
        # 3. Load the matching embedding model to transform incoming user queries
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 4. Context-anchoring prompt template
        self.prompt_template = PromptTemplate.from_template(
            "You are a professional financial analyst assistant for CrediTrust. Your task is to answer "
            "customer complaint inquiries accurately. Use ONLY the following retrieved complaint excerpts "
            "to formulate your answer. If the context doesn't contain the answer or is insufficient, state clearly "
            "that you do not have enough information to form a reliable conclusion.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def retrieve_context(self, question, k=5):
        """Calculates similarity distances between the query and the pre-computed parquet vectors."""
        # Embed the real-time user question
        query_vector = self.embedding_model.encode(question).astype('float32')
        
        # Compute cosine similarity or dot product distances across the matrix
        # Simple, fast dot product/cosine similarity check without needing FAISS overhead
        dot_products = np.dot(self.embeddings_matrix, query_vector)
        matrix_norms = np.linalg.norm(self.embeddings_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        # Handle zero-division edges safely
        norms = matrix_norms * query_norm
        norms[norms == 0] = 1e-10
        similarities = dot_products / norms
        
        # Grab top-k highest similarity index marks
        top_indices = np.argsort(similarities)[::-1][:k]
        
        retrieved_chunks = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            
            # Identify text narrative and metadata columns safely
            text_content = row.get('cleaned_narrative', row.get('text', 'No narrative text found'))
            complaint_id = row.get('complaint_id', row.get('id', 'N/A'))
            company_name = row.get('company', 'Unknown')
            category = row.get('product_category', row.get('product', 'General'))
            
            retrieved_chunks.append({
                "complaint_id": complaint_id,
                "company": company_name,
                "product_category": category,
                "text_chunk": text_content
            })
            
        return retrieved_chunks

    def generate_answer(self, question, mock_llm=False):
        """Orchestrates retrieval and generation layers."""
        # 1. Initialize variables safely so they always exist
        sources = []
        answer = ""
        
        # 2. Run context extraction
        sources = self.retrieve_context(question, k=5)
        
        # 3. Format the context string
        context_string = "\n---\n".join([
            f"[Company: {s['company']} | Product: {s['product_category']}] Narrative: {s['text_chunk']}" 
            for s in sources
        ])
        
        formatted_prompt = self.prompt_template.format(context=context_string, question=question)
        
        if mock_llm:
            return f"[Simulated Secure Analyst Response grounded on {len(sources)} sources]", sources
            
        # 4. Complete, valid Try-Except block for the LLM Layer
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                task="text-generation",
                temperature=0.1,
                max_new_tokens=256
            )
            response = llm.invoke(formatted_prompt)
            answer = response.split("Answer:")[-1].strip()
            return answer, sources
            
        except Exception as e:
            # Structurally valid fallback clause
            fallback_answer = f"System notice: Direct API connection paused. However, {len(sources)} highly relevant context rows were pulled successfully."
            return fallback_answer, sources