import streamlit as st
from src.rag_engine import CrediTrustRAG

# 1. Page Layout Configurations
st.set_page_config(page_title="CrediTrust Assistant", page_icon="🧠", layout="wide")

st.title("🧠 CrediTrust Financial Complaint RAG Assistant")
st.markdown("### Enterprise Operational Analytics & Context Retrieval Engine")
st.write("Query verified historical consumer financial data archives with zero structural hallucinations.")

# 2. Cache the RAG Backend Engine in Session State (Loads only once)
if "rag_core" not in st.session_state:
    try:
        st.session_state.rag_core = CrediTrustRAG()
    except Exception as e:
        st.error(f"Failed to load vector infrastructure components: {e}")

# 3. Initialize Conversation History Tracking State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Sidebar Control Panel
with st.sidebar:
    st.header("⚙️ Engine Configurations")
    st.info("Backend Model: `all-MiniLM-L6-v2` \n\nData Matrix: Pre-built Parquet Embeddings")
    
    # "Clear" button to reset the conversation
    if st.button("🗑️ Clear Dialogue History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. Render Active Conversational Chat History Log
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the message has sources attached, render them inside an expander
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Verified Retrieval Sources Tracked"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1}:** [ID: {src['complaint_id']}] | **Company:** {src['company']} | **Category:** {src['product_category']}")
                    st.caption(f"*Excerpt:* {src['text_chunk']}")

# 6. Capture Real-Time User Queries via Chat Input Box
if user_prompt := st.chat_input("Ask a question about credit card disputes, savings account blocks, or loan terms..."):
    
    # Display user question block instantly
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    # Process query via RAG Engine and display response
    with st.chat_message("assistant"):
        with st.spinner("Executing similarity search vector scans & compiling context..."):
            # Call your working backend engine
            answer, sources = st.session_state.rag_core.generate_answer(user_prompt, mock_llm=False)
            st.markdown(answer)
            
            # Key Usability Requirement: Render Source Cards below the generated text
            if sources:
                with st.expander("🔍 Verified Retrieval Sources Tracked"):
                    for idx, src in enumerate(sources):
                        st.markdown(f"**Source {idx+1}:** [ID: {src['complaint_id']}] | **Company:** {src['company']} | **Category:** {src['product_category']}")
                        st.caption(f"*Excerpt:* {src['text_chunk']}")
                        
        # Save the interaction records to session memory state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })