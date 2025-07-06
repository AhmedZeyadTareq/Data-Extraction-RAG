import warnings

warnings.filterwarnings("ignore",
                        message="builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute",
                        category=DeprecationWarning)

# ==== Imports ====
import os
import tempfile
import tiktoken
from PIL import Image
from markitdown import MarkItDown
from openai import OpenAI
from llama_parse import LlamaParse
import streamlit as st
import time
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re
import json

# ==== Config ====
LLAMA_API = os.getenv("LLAMA_API_PARSE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4.1-mini"

# ==== Page Configuration ====
st.set_page_config(
    page_title="Smart Content Extraction",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== Custom CSS ====
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    
    .process-step {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    
    .question-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==== Header ====
st.markdown("""
<div class="main-header">
    <h1>💡 Smart Content Extraction</h1>
    <p>🎲 Extract & derive any type of content with advanced AI techniques</p>
</div>
""", unsafe_allow_html=True)

# ==== Sidebar ====
with st.sidebar:
    # Logo section
    logo_link = "formal image.jpg"
    if os.path.exists(logo_link):
        logo_image = Image.open(logo_link)
        st.image(logo_image, width=150)
    else:
        st.warning("⚠️ Logo not found. Please check the logo path.")
    
    # Add some spacing
    st.markdown("---")
    
    # Developer info
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Eng. Ahmed Zeyad Tareq**")
    st.markdown("🎓 Master's in AI Engineering")
    st.markdown("📊 Data Scientist & AI Developer")
    
    # Add some spacing
    st.markdown("---")
    
    # Social links
    st.markdown("#### 🔗 Connect")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("[GitHub](https://github.com/AhmedZeyadTareq)")
    with col2:
        st.markdown("[LinkedIn](https://www.linkedin.com/in/ahmed-zeyad-tareq)")
    with col3:
        st.markdown("[Kaggle](https://www.kaggle.com/ahmedzeyadtareq)")
    
    # Add some spacing
    st.markdown("---")
    
    # App info
    st.markdown("#### 📊 App Statistics")
    
    # Initialize session state for statistics
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = 0
    if "questions_answered" not in st.session_state:
        st.session_state.questions_answered = 0
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Files Processed", st.session_state.files_processed)
    with col2:
        st.metric("Questions Answered", st.session_state.questions_answered)
    
    # Add some spacing
    st.markdown("---")
    
    # Additional app info
    st.markdown("#### ℹ️ About")
    st.markdown("This app extracts and analyzes content from various document formats using AI.")
    st.markdown("**Features:**")
    st.markdown("• Document extraction")
    st.markdown("• Content reorganization")
    st.markdown("• AI-powered Q&A")
    st.markdown("• Data visualization")
    
    # Add some spacing
    st.markdown("---")
    
    # Footer info
    st.markdown("#### 🚀 Tech Stack")
    st.markdown("• **Frontend:** Streamlit")
    st.markdown("• **AI:** OpenAI GPT-4")
    st.markdown("• **Parser:** LlamaParse")
    st.markdown("• **Visualization:** Plotly")
    
    # Copyright
    st.markdown("---")
    st.markdown("*© 2024 Ahmed Zeyad Tareq*")
    st.markdown("*All rights reserved*")

# ==== Main Content ====
# File Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📂 Upload Your Document")
st.markdown("**Supported formats:** PDF, DOCX, TXT, Images (PNG, JPG, JPEG), and more")

uploaded_file = st.file_uploader(
    "Choose a file to extract content from:",
    type=None,
    help="Upload any document type for intelligent content extraction"
)

if uploaded_file:
    # File info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📄 **File:** {uploaded_file.name}")
    with col2:
        st.info(f"📊 **Size:** {uploaded_file.size:,} bytes")
    with col3:
        st.info(f"🗂️ **Type:** {uploaded_file.type}")

st.markdown('</div>', unsafe_allow_html=True)

#####################################
# ==== Functions ====
#####################################

def convert_file(path: str) -> str:
    """Convert file to text (prefer structured, fallback to OCR)"""
    ext = os.path.splitext(path)[1].lower()
    try:
        print("[🔍] Trying structured text extraction via MarkItDown...")
        md = MarkItDown(enable_plugins=False)
        result = md.convert(path)
        if result.text_content.strip():
            print(f"[✔] Markdown extracted.")
            return result.text_content
        else:
            print("[⚠️] No structured text found. Fallback to OCR...")
    except Exception:
        print(f"[❌] MarkItDown failed. Fallback to OCR...")

    print("[🔍] OCR Started...")
    try:
        parser = LlamaParse(api_key=LLAMA_API, result_type="markdown")
        documents = parser.load_data(path)

        if not documents:
            st.error("Failed to parse the document - no content returned")
            return ""

        return documents[0].text

    except Exception as e:
        st.error(f"Error parsing document: {str(e)}")
        return ""

def reorganize_markdown(raw: str) -> str:
    """Reorganize markdown via OpenAI"""
    client = OpenAI()
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": f"reorganize the following content:\n {raw}"},
            {"role": "system", "content": (
                "You are a reorganizer. Return the content in Markdown, keeping it identical. "
                "Do not delete or replace anything—only reorganize for better structure. your response the content direct without (``` ```)."
            )}
        ]
    )
    print("===Reorganized Done===")
    return completion.choices[0].message.content

def rag(con: str, question: str) -> tuple[str, dict]:
    """Answer questions from provided content and detect visualization requests"""
    client = OpenAI()
    
    # Enhanced system prompt to detect visualization requests
    system_prompt = f"""You are an assistant that answers questions from provided content. 
    
    If the user asks for charts, graphs, or visualizations:
    1. First provide a text answer
    2. Then provide data in JSON format for visualization
    3. Use this format: [CHART_DATA]{{json_data}}[/CHART_DATA]
    
    For pie charts, use: {{"type": "pie", "labels": ["label1", "label2"], "values": [value1, value2], "title": "Chart Title"}}
    For bar charts, use: {{"type": "bar", "x": ["item1", "item2"], "y": [value1, value2], "title": "Chart Title"}}
    For line charts, use: {{"type": "line", "x": ["point1", "point2"], "y": [value1, value2], "title": "Chart Title"}}
    
    Answer from the following content:\n {con}"""
    
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": question},
            {"role": "system", "content": system_prompt}
        ]
    )
    
    response = completion.choices[0].message.content
    
    # Extract chart data if present
    chart_data = None
    if "[CHART_DATA]" in response and "[/CHART_DATA]" in response:
        try:
            chart_start = response.find("[CHART_DATA]") + len("[CHART_DATA]")
            chart_end = response.find("[/CHART_DATA]")
            chart_json = response[chart_start:chart_end]
            chart_data = json.loads(chart_json)
            # Remove chart data from response text
            response = response.replace(f"[CHART_DATA]{chart_json}[/CHART_DATA]", "").strip()
        except:
            chart_data = None
    
    return response, chart_data

def create_visualization(chart_data: dict, unique_key: str = None):
    """Create visualization based on chart data"""
    if not chart_data:
        return
    
    chart_type = chart_data.get("type", "").lower()
    title = chart_data.get("title", "Chart")
    
    # Generate unique key if not provided
    if unique_key is None:
        unique_key = f"chart_{int(time.time() * 1000)}"
    
    if chart_type == "pie":
        labels = chart_data.get("labels", [])
        values = chart_data.get("values", [])
        
        if labels and values:
            fig = px.pie(
                values=values,
                names=labels,
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                font=dict(size=12),
                title_font_size=16,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True, key=f"pie_{unique_key}")
    
    elif chart_type == "bar":
        x_data = chart_data.get("x", [])
        y_data = chart_data.get("y", [])
        
        if x_data and y_data:
            fig = px.bar(
                x=x_data,
                y=y_data,
                title=title,
                color=y_data,
                color_continuous_scale="viridis"
            )
            fig.update_layout(
                xaxis_title="Categories",
                yaxis_title="Values",
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True, key=f"bar_{unique_key}")
    
    elif chart_type == "line":
        x_data = chart_data.get("x", [])
        y_data = chart_data.get("y", [])
        
        if x_data and y_data:
            fig = px.line(
                x=x_data,
                y=y_data,
                title=title,
                markers=True
            )
            fig.update_layout(
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True, key=f"line_{unique_key}")

def count_tokens(content: str, model="gpt-4-turbo"):
    """Count tokens in the content"""
    enc = tiktoken.encoding_for_model(model)
    token_count = len(enc.encode(content))
    print(f"The Size of the Content_Tokens: {token_count}")
    return token_count

# ==== Main Process ====

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

        # Processing Section
        st.markdown("### 🔄 Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Extraction", type="primary", use_container_width=True):
                with st.spinner("🔍 Extracting content... This may take a moment"):
                    progress_bar = st.progress(0)
                    
                    # Simulate progress
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    raw_text = convert_file(file_path)
                    st.session_state["raw_text"] = raw_text
                    st.session_state.files_processed += 1
                    
                    # Success message
                    st.markdown("""
                    <div class="success-message">
                        ✅ <strong>Content extracted successfully!</strong><br>
                        Your document has been processed and is ready for use.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Token count
                    if raw_text:
                        token_count = count_tokens(raw_text)
                        st.info(f"📊 **Content Statistics:** {len(raw_text):,} characters, ~{token_count:,} tokens")

        with col2:
            if "raw_text" in st.session_state:
                if st.button("🧹 Reorganize Content", type="secondary", use_container_width=True):
                    with st.spinner("🔄 Reorganizing content for better structure..."):
                        progress_bar = st.progress(0)
                        
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        organized = reorganize_markdown(st.session_state["raw_text"])
                        st.session_state["organized_text"] = organized
                        
                        st.markdown("""
                        <div class="success-message">
                            ✅ <strong>Content reorganized successfully!</strong><br>
                            Your content has been restructured for better readability.
                        </div>
                        """, unsafe_allow_html=True)

        # Display extracted content
        if "raw_text" in st.session_state:
            st.markdown("### 📄 Extracted Content")
            
            # Tabs for different views
            tab1, tab2 = st.tabs(["📖 Raw Content", "✨ Organized Content"])
            
            with tab1:
                st.text_area(
                    "Raw extracted content:",
                    st.session_state["raw_text"],
                    height=300,
                    help="This is the raw content extracted from your document"
                )
            
            with tab2:
                if "organized_text" in st.session_state:
                    st.markdown(st.session_state["organized_text"])
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download Organized Content",
                        data=st.session_state["organized_text"],
                        file_name=f"organized_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.info("👆 Click 'Reorganize Content' to see the structured version")

        # Q&A Section
        if "raw_text" in st.session_state:
            st.markdown("""
            <div class="question-section">
                <h3>💬 Ask Questions About Your Content</h3>
                <p>Get instant answers from your extracted content using AI</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Question input
            question = st.text_input(
                "Ask anything about your content:",
                placeholder="e.g., What are the main topics discussed in this document?",
                help="Type your question and get AI-powered answers based on your content"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🎯 Get Answer", type="primary", use_container_width=True):
                    if question:
                        with st.spinner("🤔 Analyzing your question..."):
                            progress_bar = st.progress(0)
                            
                            for i in range(100):
                                time.sleep(0.01)
                                progress_bar.progress(i + 1)
                            
                            content_to_use = st.session_state.get("organized_text", st.session_state["raw_text"])
                            answer, chart_data = rag(content_to_use, question)
                            st.session_state.questions_answered += 1
                            
                            # Display Q&A
                            st.markdown("### 💡 AI Response")
                            
                            with st.expander("📝 Your Question", expanded=True):
                                st.markdown(f"**Q:** {question}")
                            
                            with st.expander("🤖 AI Answer", expanded=True):
                                st.markdown(f"**A:** {answer}")
                                
                                # Create visualization if chart data is present
                                if chart_data:
                                    st.markdown("### 📊 Visual Representation")
                                    create_visualization(chart_data, f"main_{int(time.time() * 1000)}")
                                
                            # Save to session state for history
                            if "qa_history" not in st.session_state:
                                st.session_state.qa_history = []
                            
                            st.session_state.qa_history.append({
                                "question": question,
                                "answer": answer,
                                "chart_data": chart_data,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    else:
                        st.warning("⚠️ Please enter a question first!")
            
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.qa_history = []
                    st.rerun()

            # Q&A History
            if "qa_history" in st.session_state and st.session_state.qa_history:
                st.markdown("### 📚 Question History")
                
                for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5 Q&As
                    with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa['question'][:50]}...", expanded=False):
                        st.markdown(f"**Question:** {qa['question']}")
                        st.markdown(f"**Answer:** {qa['answer']}")
                        
                        # Show visualization if available
                        if qa.get('chart_data'):
                            st.markdown("**📊 Visualization:**")
                            create_visualization(qa['chart_data'], f"history_{i}_{int(time.time() * 1000)}")
                        
                        st.caption(f"⏰ {qa['timestamp']}")

        # Quick Actions Section
        if "raw_text" in st.session_state:
            st.markdown("### ⚡ Quick Actions")
            st.markdown("Try these common questions:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Create Summary Chart", use_container_width=True):
                    st.session_state.quick_question = "Create a pie chart showing the main topics or categories in this content"
                    st.rerun()
            
            with col2:
                if st.button("📈 Show Data Trends", use_container_width=True):
                    st.session_state.quick_question = "Create a bar chart showing any numerical data or statistics from this content"
                    st.rerun()
            
            with col3:
                if st.button("🔍 Key Insights", use_container_width=True):
                    st.session_state.quick_question = "What are the key insights and main points from this content?"
                    st.rerun()
            
            # Handle quick questions
            if "quick_question" in st.session_state:
                question = st.session_state.quick_question
                del st.session_state.quick_question
                
                with st.spinner("🤔 Processing your quick question..."):
                    content_to_use = st.session_state.get("organized_text", st.session_state["raw_text"])
                    answer, chart_data = rag(content_to_use, question)
                    st.session_state.questions_answered += 1
                    
                    st.markdown("### 💡 Quick Answer")
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")
                    
                    if chart_data:
                        st.markdown("### 📊 Visual Representation")
                        create_visualization(chart_data, f"quick_{int(time.time() * 1000)}")
                    
                    # Save to history
                    if "qa_history" not in st.session_state:
                        st.session_state.qa_history = []
                    
                    st.session_state.qa_history.append({
                        "question": question,
                        "answer": answer,
                        "chart_data": chart_data,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

# ==== Footer ====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 Built with Streamlit | 💡 Powered by OpenAI & LlamaParse | 🎯 Smart Content Extraction</p>
    <p><small>© 2024 Ahmed Zeyad Tareq - All rights reserved</small></p>
</div>
""", unsafe_allow_html=True)
