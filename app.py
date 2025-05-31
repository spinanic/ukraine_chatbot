import streamlit as st
import os

# Page config
st.set_page_config(page_title="Ukraine Conflict Chatbot", page_icon="🇺🇦")
st.title("🇺🇦 Ukraine Conflict Analysis Chatbot")

# Check for API key from secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    api_available = True
except:
    api_available = False
    st.error("No API key configured. This is a demo version.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📁 File Upload")
    
    # File uploader (max 200MB as per Streamlit limits)
    uploaded_files = st.file_uploader(
        "Upload PDFs or CSVs (max 200MB)",
        type=['pdf', 'csv', 'txt'],
        accept_multiple_files=True,
        help="Note: Files over 100MB may cause issues with GitHub"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size / 1024 / 1024:.1f} MB)")
    
    st.markdown("---")
    st.info("This chatbot analyzes documents about the Ukraine conflict")

# Main chat area
st.markdown("### 💬 Chat")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the Ukraine conflict..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if api_available:
            try:
                # Import libraries only if API is available
                from llama_index.core import Document, VectorStoreIndex
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                from llama_index.llms.groq import Groq
                from llama_index.core import Settings
                
                # Setup models
                embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                Settings.embed_model = embed_model
                
                llm = Groq(
                    model="mixtral-8x7b-32768",
                    api_key=GROQ_API_KEY
                )
                Settings.llm = llm
                
                # Create simple index with demo content
                demo_doc = Document(text="""
                The Ukraine conflict began with the 2014 annexation of Crimea. 
                In 2022, Russia launched a full-scale invasion. 
                Ukraine has shown remarkable resilience.
                International support includes military aid and sanctions.
                """)
                
                index = VectorStoreIndex.from_documents([demo_doc])
                query_engine = index.as_query_engine()
                
                response = query_engine.query(prompt)
                response_text = str(response.response)
                
            except Exception as e:
                response_text = f"Error: {str(e)}"
        else:
            response_text = "API not configured. This is a demo response. Please deploy with proper API configuration."
        
        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Footer
st.markdown("---")
st.caption("Educational tool for Ukraine conflict analysis")
