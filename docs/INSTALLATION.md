# 📦 Installation Guide

Complete guide for installing Enterprise AI Assistant dependencies

---

## 🎯 Choose Your Installation Type

### **Option 1: Full Installation** (Recommended for Development)
Everything included - all features, development tools, testing frameworks

```bash
pip install -r requirements_complete.txt
pip install -e .
```

**What you get:**
- ✅ All core features
- ✅ Document processing (all formats)
- ✅ API server
- ✅ Streamlit dashboard
- ✅ NLP capabilities
- ✅ AWS integration
- ✅ Testing tools

**Disk space**: ~1.5 GB  
**Installation time**: 5-10 minutes

---

### **Option 2: Minimal Installation** (For Production/Lambda)
Core features only - smallest footprint

```bash
pip install -r requirements_minimal.txt
```

**What you get:**
- ✅ Core RAG functionality
- ✅ API server
- ✅ Vector search
- ✅ LLM integration
- ❌ Document processing
- ❌ Dashboard
- ❌ Advanced NLP

**Disk space**: ~500 MB  
**Installation time**: 2-3 minutes

---

### **Option 3: Development Installation** (For Contributors)
Full installation + development tools

```bash
pip install -r requirements_dev.txt
pip install -e .
```

**What you get:**
- ✅ Everything from full installation
- ✅ Testing frameworks (pytest)
- ✅ Code formatters (black, isort)
- ✅ Linters (flake8, pylint, mypy)
- ✅ Jupyter notebooks
- ✅ Debugging tools
- ✅ Documentation generators

**Disk space**: ~2 GB  
**Installation time**: 10-15 minutes

---

## 🔧 Step-by-Step Installation

### **Prerequisites**

1. **Python 3.10 or higher**
   ```bash
   python --version
   # Should show: Python 3.10.x or higher
   ```

2. **pip (Python package manager)**
   ```bash
   pip --version
   # Should show: pip 23.x or higher
   ```

3. **Git** (for cloning repository)
   ```bash
   git --version
   ```

---

### **Step 1: Clone Repository**

```bash
git clone https://github.com/yourusername/enterprise-ai-assistant.git
cd enterprise-ai-assistant
```

---

### **Step 2: Create Virtual Environment**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### **Step 3: Upgrade pip**

```bash
pip install --upgrade pip
```

---

### **Step 4: Install Dependencies**

**Choose your installation type:**

```bash
# Full installation (recommended)
pip install -r requirements_complete.txt

# OR Minimal installation
pip install -r requirements_minimal.txt

# OR Development installation
pip install -r requirements_dev.txt
```

---

### **Step 5: Install Package in Editable Mode**

```bash
pip install -e .
```

This allows you to import `src` modules from anywhere.

---

### **Step 6: Configure Environment Variables**

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your credentials
# Required: OPENAI_API_KEY
```

**Minimum .env configuration:**
```env
OPENAI_API_KEY=your_openai_key_here
APP_NAME=Enterprise AI Assistant
LOG_LEVEL=INFO
```

---

### **Step 7: Verify Installation**

```bash
# Test imports
python -c "from src.api.api_app import app; print('✓ Imports working!')"

# Run a quick test
python tests/integration/test_phase5_nlp.py
```

If no errors, installation is successful! ✅

---

## 🐍 Installing Individual Components

If you want to install specific components only:

### **API Server Only**
```bash
pip install fastapi uvicorn[standard] pydantic python-multipart openai langchain faiss-cpu numpy
```

### **Document Processing Only**
```bash
pip install PyPDF2 pdfplumber python-docx openpyxl pandas Pillow nltk
```

### **Dashboard Only**
```bash
pip install streamlit plotly pandas requests
```

### **NLP Features Only**
```bash
pip install openai tiktoken langchain
```

---

## 📋 Requirements Files Explained

### **requirements_complete.txt**
- All production dependencies
- Document processing libraries
- API framework
- Dashboard
- NLP tools
- AWS integration
- **Use for**: Development, full features

### **requirements_minimal.txt**
- Core dependencies only
- No document processing
- No dashboard
- API + RAG only
- **Use for**: Production deployment, AWS Lambda

### **requirements_dev.txt**
- Everything from complete
- Testing frameworks
- Code quality tools
- Development utilities
- **Use for**: Contributing, development

---

## 🔍 Troubleshooting

### **Issue: pip install fails**

**Solution 1**: Update pip
```bash
pip install --upgrade pip setuptools wheel
```

**Solution 2**: Install with no cache
```bash
pip install --no-cache-dir -r requirements_complete.txt
```

**Solution 3**: Install one by one
```bash
pip install openai
pip install langchain
# ... continue for each package
```

---

### **Issue: FAISS installation fails on Windows**

**Solution**: Use pre-built wheel
```bash
pip install faiss-cpu --no-cache-dir
```

---

### **Issue: "No module named 'src'"**

**Solution**: Install package in editable mode
```bash
pip install -e .
```

---

### **Issue: Import errors after installation**

**Solution**: Verify virtual environment is activated
```bash
# Should show (venv) in prompt
# If not:
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

---

### **Issue: Memory error during installation**

**Solution**: Install with lower memory usage
```bash
pip install --no-cache-dir -r requirements_complete.txt
```

---

### **Issue: SSL certificate error**

**Solution**: Use trusted host
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements_complete.txt
```

---

## 📊 Package Sizes

| Package | Size | Purpose |
|---------|------|---------|
| OpenAI | 50 MB | LLM integration |
| LangChain | 100 MB | RAG framework |
| FAISS | 200 MB | Vector search |
| Transformers | 300 MB | NLP models (optional) |
| FastAPI | 50 MB | API framework |
| Streamlit | 150 MB | Dashboard |
| PyTorch | 800 MB | Deep learning (optional) |

**Total (full install)**: ~1.5 GB

---

## ⚡ Quick Start After Installation

### **1. Start API Server**
```bash
python -m src.api.api_app
```
Visit: http://localhost:8000/docs

### **2. Start Dashboard**
```bash
streamlit run streamlit_app.py
```
Visit: http://localhost:8501

### **3. Run Tests**
```bash
python test_api_endpoints.py
```

---

## 🆘 Getting Help

If installation issues persist:

1. **Check Python version**: Must be 3.10+
2. **Check pip version**: Should be 23.0+
3. **Try minimal install first**: `requirements_minimal.txt`
4. **Check system resources**: Need ~2 GB free space
5. **Check internet connection**: Some packages are large

---

## 🎯 Platform-Specific Notes

### **Windows**
- Use PowerShell or Command Prompt
- May need Visual C++ Build Tools for some packages
- Activate venv: `venv\Scripts\activate`

### **macOS**
- May need Xcode Command Line Tools: `xcode-select --install`
- Activate venv: `source venv/bin/activate`
- M1/M2 Macs: Some packages may need Rosetta 2

### **Linux**
- May need build essentials: `sudo apt-get install build-essential`
- Activate venv: `source venv/bin/activate`
- Ubuntu/Debian: `sudo apt-get install python3-dev`

---

## ✅ Verification Checklist

After installation, verify:

- [ ] Python version 3.10+
- [ ] Virtual environment activated
- [ ] All packages installed: `pip list`
- [ ] .env file configured with API keys
- [ ] Package installed in editable mode: `pip show enterprise-ai-assistant`
- [ ] Imports working: `python -c "from src.api.api_app import app"`
- [ ] Tests passing: Run at least one test file

---

## 🚀 Next Steps

After successful installation:

1. ✅ Review [README.md](../README.md) for usage examples
2. ✅ Check [API Documentation](http://localhost:8000/docs)
3. ✅ Try the [Streamlit Dashboard](http://localhost:8501)
4. ✅ Read [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production setup

---

**Installation complete!** 🎉

You're ready to start using Enterprise AI Assistant!