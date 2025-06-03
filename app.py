import streamlit as st
import os
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Configure page
st.set_page_config(page_title="Ukraine Conflict Chatbot", page_icon="🇺🇦")
st.title("🇺🇦 Ukraine Conflict Analysis Chatbot")

# Load ACLED data
@st.cache_data
def load_acled_data():
    return pd.read_csv("data/acled_ukraine_filtered.csv", parse_dates=["event_date"])

acled_df = load_acled_data()

# API Key Check
try:
    api_key = os.environ["OPENAI_API_KEY"]
    api_available = True
except KeyError:
    st.error("No API key found in environment. Set OPENAI_API_KEY first.")
    api_available = False

# Sidebar
with st.sidebar:
    st.header("📁 File Upload")
    uploaded_files = st.file_uploader(
        "Upload PDFs or CSVs (max 200MB)",
        type=['pdf', 'csv', 'txt'],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size / 1024 / 1024:.1f} MB)")

    st.markdown("---")
    st.info("This chatbot analyzes ISW reports and ACLED battle data.")
    st.markdown("---")
    st.write(f"ACLED rows: {acled_df.shape[0]}")

    if st.checkbox("Show ACLED sample"):
        st.dataframe(acled_df.head(10), height=200)

    if st.checkbox("Plot battle event counts by month"):
        battles = acled_df[acled_df["event_type"] == "Battle"]
        battles["month"] = battles["event_date"].dt.to_period("M")
        monthly_counts = battles.groupby("month").size().reset_index(name="count")
        monthly_counts["month"] = monthly_counts["month"].dt.to_timestamp()
        st.line_chart(monthly_counts.set_index("month")["count"])

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
st.markdown("### 💬 Chat")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the Ukraine conflict..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if api_available:
        try:
            # Setup models
            embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
            Settings.llm = llm
            Settings.embed_model = embed_model

            # Load stored index
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            index = load_index_from_storage(storage_context)
            query_engine = index.as_query_engine()
            response = query_engine.query(prompt)
            response_text = str(response)

        except Exception as e:
            response_text = f"Error: {e}"
    else:
        response_text = "API key missing. Cannot generate response."

    st.chat_message("assistant").markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Footer
st.markdown("---")
st.caption("Educational tool for Ukraine conflict analysis")

