# 🚀 Quick Reference Card

One-page reference for Enterprise AI Assistant

---

## 📦 Installation (One Command)

```bash
pip install -r requirements_complete.txt && pip install -e .
```

---

## ⚡ Quick Start

```bash
# 1. Configure
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 2. Start API
python -m src.api.api_app

# 3. Start Dashboard (new terminal)
streamlit run streamlit_app.py
```

---

## 🔗 URLs

- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

---

## 📝 Common Commands

### **API Testing**
```bash
# Test all endpoints
python test_api_endpoints.py

# Test single endpoint
curl http://localhost:8000/health
```

### **Run Tests**
```bash
python tests/integration/test_phase5_nlp.py
```

### **Docker**
```bash
# Build
docker build -t enterprise-ai -f deployment/docker/Dockerfile .

# Run
docker run -d -p 8000:8000 --env-file .env enterprise-ai

# With Docker Compose
cd deployment/docker && docker-compose up -d
```

---

## 🔧 Python Usage

### **Query Knowledge Base**
```python
from src.llm.enhanced_rag_pipeline import EnhancedRAGPipeline

response = pipeline.query(
    question="Your question here",
    retrieval_method="multi_query",  # simple, multi_query, self_rag, hybrid
    optimize_for="balanced"          # cost, speed, quality, balanced
)
print(response['answer'])
```

### **NLP Analysis**
```python
from src.nlp.nlp_analyzer import NLPAnalyzer

analyzer = NLPAnalyzer(openai_api_key="your_key")

# Sentiment
sentiment = analyzer.sentiment_analysis("Great product!")

# NER
entities = analyzer.named_entity_recognition("John works at Microsoft")

# Summarize
summary = analyzer.summarize_text(long_text, max_length=150)
```

### **Process Documents**
```python
from src.ingestion.document_processor import DocumentProcessor

processor = DocumentProcessor(processed_dir="data/processed")
chunks = processor.process_file("document.pdf")
```

---

## 🌐 API Endpoints

### **Query**
```bash
POST /query
{
  "question": "What is machine learning?",
  "retrieval_method": "multi_query",
  "k": 5,
  "optimize_for": "balanced"
}
```

### **NLP**
```bash
POST /nlp/sentiment  # Sentiment analysis
POST /nlp/ner        # Named entities
POST /nlp/classify   # Classification
POST /nlp/summarize  # Summarization
```

### **Analytics**
```bash
GET /analytics/summary     # Analytics summary
GET /analytics/cost?days=30  # Cost report
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/api/api_app.py` | REST API |
| `streamlit_app.py` | Dashboard |
| `src/llm/enhanced_rag_pipeline.py` | Main RAG logic |
| `src/nlp/nlp_analyzer.py` | NLP features |
| `.env` | Configuration |
| `requirements_complete.txt` | All dependencies |

---

## 🎯 Retrieval Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `simple` | Single query | Fast, basic |
| `multi_query` | Multiple variations | Better coverage |
| `self_rag` | Self-evaluating | Highest quality |
| `hybrid` | Semantic + keyword | Best accuracy |

---

## 💰 Optimization Strategies

| Strategy | Speed | Cost | Quality |
|----------|-------|------|---------|
| `cost` | Medium | 💰 | Good |
| `speed` | ⚡⚡⚡ | 💰💰 | Good |
| `quality` | Slow | 💰💰💰 | ⭐⭐⭐ |
| `balanced` | Fast | 💰💰 | ⭐⭐ |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import error | `pip install -e .` |
| API not starting | Check `.env` has `OPENAI_API_KEY` |
| Module not found | Activate venv: `venv\Scripts\activate` |
| Port already in use | Change port or kill process |

---

## 📊 Environment Variables

```env
# Required
OPENAI_API_KEY=sk-...

# Optional
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-bucket

# Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
```

---

## 🚀 Deployment

### **AWS Lambda**
```bash
cd deployment/aws
sam build
sam deploy --guided
```

### **Docker**
```bash
docker build -t enterprise-ai .
docker run -p 8000:8000 --env-file .env enterprise-ai
```

---

## 📈 Monitoring

```python
# Get analytics
from src.analytics.analytics_tracker import AnalyticsTracker

tracker = AnalyticsTracker()
analytics = tracker.get_query_analytics(days=7)

print(f"Queries: {analytics['total_queries']}")
print(f"Cost: ${analytics['total_cost']:.2f}")
```

---

## 🔑 Key Features

✅ Multi-format document processing  
✅ Advanced RAG (4 strategies)  
✅ Multi-LLM support  
✅ NLP capabilities  
✅ REST API  
✅ Interactive dashboard  
✅ Cost optimization  
✅ Analytics tracking  
✅ Docker ready  
✅ AWS Lambda ready  

---

## 📞 Quick Help

```bash
# Check version
python --version  # Need 3.10+

# List installed packages
pip list | grep -E "openai|langchain|faiss|fastapi|streamlit"

# Test API is running
curl http://localhost:8000/health

# View logs
tail -f logs/phase5_test.log
```

---

## 💡 Tips

- **Start simple**: Use `simple` retrieval first
- **Optimize costs**: Use `cost` optimization for dev
- **Test locally**: Before deploying to cloud
- **Monitor costs**: Check `/analytics/cost` regularly
- **Use dashboard**: Easier than API for testing

---

**Keep this card handy for quick reference!** 📌

Print or save as PDF for offline access.