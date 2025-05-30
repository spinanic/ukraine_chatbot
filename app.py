import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import tempfile
import shutil

# Page config
st.set_page_config(
    page_title="Ukraine Conflict Analysis Hub", 
    page_icon="🇺🇦",
    layout="wide"
)

# Title and description
st.title("🇺🇦 Ukraine Conflict Analysis Hub")
st.markdown("Analyze ISW reports, ACLED data, and visualize conflict trends")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "data" not in st.session_state:
    st.session_state.data = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Get API key from Streamlit secrets (for deployment)
# You'll set this in Streamlit Cloud settings
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("No API key found. Please set GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# Configure models
@st.cache_resource
def setup_models():
    # Embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.embed_model = embed_model
    
    # LLM
    llm = Groq(
        model="mixtral-8x7b-32768",
        api_key=GROQ_API_KEY,
        temperature=0.1
    )
    Settings.llm = llm
    
    return True

# Data visualization functions
def create_timeline_plot(df, date_col, event_col):
    """Create timeline visualization of events"""
    fig = px.scatter(
        df, 
        x=date_col, 
        y=event_col,
        title="Conflict Timeline",
        hover_data=df.columns
    )
    return fig

def create_heatmap(df, lat_col, lon_col, intensity_col):
    """Create geographic heatmap"""
    fig = px.density_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        z=intensity_col,
        radius=10,
        center=dict(lat=48.3794, lon=31.1656),  # Center on Ukraine
        zoom=5,
        mapbox_style="open-street-map",
        title="Conflict Intensity Heatmap"
    )
    return fig

def create_bar_chart(df, x_col, y_col, title="Analysis"):
    """Create bar chart for categorical analysis"""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=y_col,
        color_continuous_scale="Reds"
    )
    return fig

def analyze_csv_data(file_content, file_name):
    """Analyze uploaded CSV data"""
    try:
        df = pd.read_csv(file_content)
        st.session_state.data[file_name] = df
        
        # Basic analysis
        analysis = {
            "rows": len(df),
            "columns": list(df.columns),
            "date_columns": [col for col in df.columns if 'date' in col.lower()],
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "summary": df.describe().to_dict()
        }
        
        return analysis, df
    except Exception as e:
        return {"error": str(e)}, None

# Visualization tool for LLM
def create_visualization(query: str) -> str:
    """Tool for LLM to create visualizations"""
    try:
        # Parse the query to understand what visualization is needed
        query_lower = query.lower()
        
        # Get available data
        available_data = list(st.session_state.data.keys())
        
        if not available_data:
            return "No data available for visualization. Please upload CSV files first."
        
        # Use the first available dataset
        df = st.session_state.data[available_data[0]]
        
        # Determine visualization type
        if "timeline" in query_lower or "time" in query_lower:
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                fig = create_timeline_plot(df, date_cols[0], df.columns[1])
                st.plotly_chart(fig)
                return f"Created timeline visualization using {date_cols[0]}"
        
        elif "map" in query_lower or "geographic" in query_lower:
            lat_cols = [col for col in df.columns if 'lat' in col.lower()]
            lon_cols = [col for col in df.columns if 'lon' in col.lower()]
            if lat_cols and lon_cols:
                fig = create_heatmap(df, lat_cols[0], lon_cols[0], df.columns[-1])
                st.plotly_chart(fig)
                return "Created geographic heatmap"
        
        elif "bar" in query_lower or "compare" in query_lower:
            if len(df.columns) >= 2:
                fig = create_bar_chart(df, df.columns[0], df.columns[1])
                st.plotly_chart(fig)
                return f"Created bar chart comparing {df.columns[0]} and {df.columns[1]}"
        
        # Default: show data summary
        st.dataframe(df.head(10))
        return f"Displayed data preview. Dataset has {len(df)} rows and {len(df.columns)} columns."
        
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

# Setup models
setup_models()

# Create visualization tool for LLM
viz_tool = FunctionTool.from_defaults(
    fn=create_visualization,
    name="create_visualization",
    description="Create data visualizations including timelines, maps, and charts"
)

# Sidebar for file upload and data management
with st.sidebar:
    st.header("📁 Data Management")
    
    # File upload
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDFs, CSVs, or text files",
        type=['pdf', 'csv', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(file.name)
                
                # Process CSV files
                if file.name.endswith('.csv'):
                    analysis, df = analyze_csv_data(file, file.name)
                    if df is not None:
                        st.success(f"✅ Loaded {file.name}")
                        with st.expander(f"Preview {file.name}"):
                            st.write(f"Shape: {df.shape}")
                            st.dataframe(df.head())
    
    # Show loaded data
    st.subheader("📊 Loaded Datasets")
    for name in st.session_state.data.keys():
        st.write(f"• {name}")
    
    # Quick visualizations
    if st.session_state.data:
        st.subheader("🎨 Quick Visualizations")
        dataset = st.selectbox("Choose dataset:", list(st.session_state.data.keys()))
        viz_type = st.selectbox("Visualization type:", ["Preview", "Timeline", "Bar Chart", "Heatmap"])
        
        if st.button("Generate"):
            df = st.session_state.data[dataset]
            
            if viz_type == "Preview":
                st.dataframe(df.head(10))
            elif viz_type == "Timeline":
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                if date_cols:
                    fig = create_timeline_plot(df, date_cols[0], df.columns[1])
                    st.plotly_chart(fig)
            elif viz_type == "Bar Chart":
                if len(df.columns) >= 2:
                    fig = create_bar_chart(df, df.columns[0], df.columns[1])
                    st.plotly_chart(fig)

# Main chat interface
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Data Analysis", "📚 Documents"])

with tab1:
    # Load or create index
    @st.cache_resource
    def load_index(force_reload=False):
        try:
            docs = []
            
            # Add uploaded PDFs
            for file in uploaded_files:
                if file.name.endswith('.pdf'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    
                    pdf_docs = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
                    docs.extend(pdf_docs)
                    os.unlink(tmp_path)
            
            # Add some default content if no docs
            if not docs:
                default_text = """
                Ukraine Conflict Information Hub
                
                This system can analyze:
                - ISW (Institute for the Study of War) reports
                - ACLED (Armed Conflict Location & Event Data)
                - News and analysis documents
                - Statistical data about the conflict
                
                Upload PDFs or CSV files to begin analysis.
                You can ask questions about the conflict, request visualizations,
                and analyze trends in the data.
                """
                docs = [Document(text=default_text)]
            
            # Create index with tools
            index = VectorStoreIndex.from_documents(docs)
            
            return index
            
        except Exception as e:
            st.error(f"Error creating index: {str(e)}")
            return None
    
    # Load index
    if st.session_state.index is None or st.button("🔄 Reload Index"):
        with st.spinner("Loading documents..."):
            st.session_state.index = load_index(force_reload=True)
    
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the conflict or request a visualization..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                # Check if visualization is requested
                viz_keywords = ['graph', 'chart', 'plot', 'visualize', 'show', 'map', 'timeline']
                wants_viz = any(keyword in prompt.lower() for keyword in viz_keywords)
                
                if wants_viz and st.session_state.data:
                    # Create visualization
                    result = create_visualization(prompt)
                    st.write(result)
                    response = f"I've created a visualization based on your request. {result}"
                else:
                    # Regular Q&A
                    query_engine = st.session_state.index.as_query_engine(
                        tools=[viz_tool],
                        similarity_top_k=3
                    )
                    response = query_engine.query(prompt)
                    response = str(response.response)
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                if "rate limit" in str(e).lower():
                    st.info("Rate limit reached. Please wait a moment.")

with tab2:
    st.header("📊 Data Analysis Dashboard")
    
    if st.session_state.data:
        # Data selection
        selected_data = st.selectbox(
            "Select dataset to analyze:",
            list(st.session_state.data.keys())
        )
        
        if selected_data:
            df = st.session_state.data[selected_data]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Info")
                st.write(f"**Rows:** {len(df)}")
                st.write(f"**Columns:** {len(df.columns)}")
                st.write("**Column Names:**")
                st.write(list(df.columns))
            
            with col2:
                st.subheader("Data Types")
                st.write(df.dtypes)
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(20))
            
            # Basic statistics
            st.subheader("Statistical Summary")
            st.write(df.describe())
            
            # Custom analysis
            st.subheader("Custom Analysis")
            col_x = st.selectbox("X-axis:", df.columns)
            col_y = st.selectbox("Y-axis:", df.columns)
            chart_type = st.selectbox("Chart type:", ["scatter", "bar", "line", "box"])
            
            if st.button("Create Chart"):
                if chart_type == "scatter":
                    fig = px.scatter(df, x=col_x, y=col_y)
                elif chart_type == "bar":
                    fig = px.bar(df, x=col_x, y=col_y)
                elif chart_type == "line":
                    fig = px.line(df, x=col_x, y=col_y)
                elif chart_type == "box":
                    fig = px.box(df, x=col_x, y=col_y)
                
                st.plotly_chart(fig)
    else:
        st.info("Upload CSV files to begin data analysis")

with tab3:
    st.header("📚 Document Library")
    st.write("Uploaded documents:")
    
    for file_name in st.session_state.uploaded_files:
        st.write(f"• {file_name}")
    
    if not st.session_state.uploaded_files:
        st.info("No documents uploaded yet. Use the sidebar to upload PDFs, CSVs, or text files.")

# Footer
st.markdown("---")
st.markdown("🎓 Educational tool for analyzing the Ukraine conflict using ISW reports and ACLED data")
