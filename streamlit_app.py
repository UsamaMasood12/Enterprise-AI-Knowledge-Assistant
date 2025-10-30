"""
Streamlit Dashboard
Interactive Frontend for Enterprise AI Assistant
"""

import streamlit as st
import requests
from typing import Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Enterprise AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def query_api(question: str, method: str = "multi_query", optimize: str = "balanced") -> Dict[str, Any]:
    """Query the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "retrieval_method": method,
                "k": 5,
                "optimize_for": optimize,
                "include_sources": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_analytics() -> Dict[str, Any]:
    """Get analytics data"""
    try:
        response = requests.get(f"{API_BASE_URL}/analytics/summary", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/nlp/sentiment",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def extract_entities(text: str) -> Dict[str, Any]:
    """Extract named entities"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/nlp/ner",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def summarize_text(text: str, max_length: int = 150) -> Dict[str, Any]:
    """Summarize text"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/nlp/summarize",
            json={"text": text, "max_length": max_length},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Enterprise AI Assistant</h1>', unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_health()
    
    if not api_status:
        st.error("⚠️ API is not running! Please start the API server first.")
        st.code("python -m src.api.api_app", language="bash")
        return
    
    st.success("✅ API Connected")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        page = st.radio(
            "Navigate to:",
            ["💬 Chat Assistant", "📊 Analytics Dashboard", "🔍 NLP Tools", "📄 Document Upload"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Query settings (for chat page)
        if page == "💬 Chat Assistant":
            st.subheader("Query Settings")
            
            retrieval_method = st.selectbox(
                "Retrieval Method",
                ["simple", "multi_query", "self_rag", "hybrid"],
                index=1
            )
            
            optimization = st.selectbox(
                "Optimization",
                ["cost", "speed", "quality", "balanced"],
                index=3
            )
        
        st.divider()
        st.caption("Enterprise AI Assistant v1.0")
    
    # Main content based on page
    if page == "💬 Chat Assistant":
        show_chat_page(retrieval_method, optimization)
    
    elif page == "📊 Analytics Dashboard":
        show_analytics_page()
    
    elif page == "🔍 NLP Tools":
        show_nlp_page()
    
    elif page == "📄 Document Upload":
        show_upload_page()


def show_chat_page(retrieval_method: str, optimization: str):
    """Chat interface page"""
    
    st.header("💬 Chat with Your Knowledge Base")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "metadata" in message and message["role"] == "assistant":
                with st.expander("📊 Query Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{message['metadata'].get('response_time', 0):.2f}s")
                    with col2:
                        st.metric("Cost", f"${message['metadata'].get('estimated_cost', 0):.4f}")
                    with col3:
                        st.metric("Sources", message['metadata'].get('num_sources', 0))
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_api(prompt, retrieval_method, optimization)
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    answer = response.get("answer", "No answer received")
                    st.markdown(answer)
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": response
                    })
                    
                    # Show details
                    with st.expander("📊 Query Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Response Time", f"{response.get('response_time', 0):.2f}s")
                        with col2:
                            st.metric("Cost", f"${response.get('estimated_cost', 0):.4f}")
                        with col3:
                            st.metric("Sources", response.get('num_sources', 0))
                        
                        st.caption(f"Provider: {response.get('llm_provider', 'Unknown')}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


def show_analytics_page():
    """Analytics dashboard page"""
    
    st.header("📊 Analytics Dashboard")
    
    # Fetch analytics
    analytics = get_analytics()
    
    if not analytics or "error" in analytics:
        st.warning("No analytics data available yet. Start using the chat to generate data!")
        return
    
    # Key metrics
    st.subheader("📈 Key Metrics")
    
    query_analytics = analytics.get("query_analytics", {})
    cost_report = analytics.get("cost_report", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Queries",
            query_analytics.get("total_queries", 0)
        )
    
    with col2:
        st.metric(
            "Avg Response Time",
            f"{query_analytics.get('avg_response_time', 0):.2f}s"
        )
    
    with col3:
        st.metric(
            "Total Cost",
            f"${query_analytics.get('total_cost', 0):.2f}"
        )
    
    with col4:
        st.metric(
            "Avg Cost/Query",
            f"${query_analytics.get('avg_cost_per_query', 0):.4f}"
        )
    
    st.divider()
    
    # Provider breakdown
    if "by_provider" in query_analytics:
        st.subheader("🤖 Usage by Provider")
        
        provider_data = query_analytics["by_provider"]
        
        if provider_data:
            df_providers = pd.DataFrame([
                {"Provider": k, "Queries": v["count"], "Cost": v["cost"]}
                for k, v in provider_data.items()
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    df_providers,
                    values="Queries",
                    names="Provider",
                    title="Queries by Provider"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    df_providers,
                    x="Provider",
                    y="Cost",
                    title="Cost by Provider"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Popular queries
    popular_queries = analytics.get("popular_queries", [])
    
    if popular_queries:
        st.subheader("🔥 Top Queries")
        
        df_popular = pd.DataFrame(popular_queries)
        st.dataframe(
            df_popular,
            use_container_width=True,
            hide_index=True
        )
    
    # Cost breakdown
    if cost_report and "daily_costs" in cost_report:
        st.divider()
        st.subheader("💰 Daily Cost Breakdown")
        
        daily_costs = cost_report["daily_costs"]
        
        if daily_costs:
            df_daily = pd.DataFrame([
                {"Date": k, "Cost": v}
                for k, v in daily_costs.items()
            ])
            
            fig = px.line(
                df_daily,
                x="Date",
                y="Cost",
                title="Daily API Costs",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)


def show_nlp_page():
    """NLP tools page"""
    
    st.header("🔍 NLP Analysis Tools")
    
    tab1, tab2, tab3 = st.tabs(["😊 Sentiment Analysis", "🏷️ Named Entities", "📝 Summarization"])
    
    # Sentiment Analysis
    with tab1:
        st.subheader("Sentiment Analysis")
        
        sentiment_text = st.text_area(
            "Enter text to analyze sentiment:",
            height=150,
            placeholder="Type or paste text here..."
        )
        
        if st.button("Analyze Sentiment", key="sentiment_btn"):
            if sentiment_text:
                with st.spinner("Analyzing..."):
                    result = analyze_sentiment(sentiment_text)
                    
                    if result and "sentiment" in result:
                        sentiment = result["sentiment"]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Sentiment", sentiment.get("label", "Unknown"))
                        
                        with col2:
                            st.metric("Confidence", f"{sentiment.get('score', 0):.2%}")
                        
                        if "explanation" in sentiment:
                            st.info(f"💡 {sentiment['explanation']}")
                    else:
                        st.error("Failed to analyze sentiment")
            else:
                st.warning("Please enter some text")
    
    # Named Entity Recognition
    with tab2:
        st.subheader("Named Entity Recognition")
        
        ner_text = st.text_area(
            "Enter text to extract entities:",
            height=150,
            placeholder="Type or paste text here..."
        )
        
        if st.button("Extract Entities", key="ner_btn"):
            if ner_text:
                with st.spinner("Extracting..."):
                    result = extract_entities(ner_text)
                    
                    if result and "entities" in result:
                        entities = result["entities"]
                        
                        if entities:
                            for entity_type, items in entities.items():
                                if items:
                                    st.subheader(f"🏷️ {entity_type}")
                                    
                                    if isinstance(items[0], dict):
                                        st.write(", ".join([item["text"] for item in items]))
                                    else:
                                        st.write(", ".join(items))
                        else:
                            st.info("No entities found")
                    else:
                        st.error("Failed to extract entities")
            else:
                st.warning("Please enter some text")
    
    # Summarization
    with tab3:
        st.subheader("Text Summarization")
        
        summ_text = st.text_area(
            "Enter text to summarize:",
            height=200,
            placeholder="Type or paste long text here..."
        )
        
        max_length = st.slider("Summary length (words)", 50, 300, 150)
        
        if st.button("Summarize", key="summ_btn"):
            if summ_text:
                with st.spinner("Summarizing..."):
                    result = summarize_text(summ_text, max_length)
                    
                    if result and "summary" in result:
                        summary_data = result["summary"]
                        
                        st.success("✅ Summary:")
                        st.write(summary_data.get("summary", ""))
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Original", f"{summary_data.get('original_length', 0)} words")
                        
                        with col2:
                            st.metric("Summary", f"{summary_data.get('summary_length', 0)} words")
                        
                        with col3:
                            st.metric("Compression", f"{summary_data.get('compression_ratio', 0):.1%}")
                    else:
                        st.error("Failed to summarize")
            else:
                st.warning("Please enter some text")


def show_upload_page():
    """Document upload page"""
    
    st.header("📄 Document Upload")
    
    st.info("⚠️ Document upload requires document processor. Currently disabled in API.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        help="Upload documents to add to the knowledge base"
    )
    
    if uploaded_file:
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        st.write(f"**Type:** {uploaded_file.type}")
        
        if st.button("📤 Upload & Process"):
            st.warning("Document processing will be implemented when vector store has data.")
            # Future: Implement actual upload
            # with st.spinner("Processing document..."):
            #     files = {"file": uploaded_file}
            #     response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)


if __name__ == "__main__":
    main()