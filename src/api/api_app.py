"""
FastAPI Application
REST API for Enterprise AI Assistant
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from loguru import logger
import uvicorn
from datetime import datetime

from ..embeddings.embeddings_generator import EmbeddingsGenerator
from ..embeddings.faiss_store import FAISSVectorStore
from ..retrieval.rag_retriever import AdvancedRAGRetriever
from ..llm.multi_llm_manager import MultiLLMManager
from ..llm.enhanced_rag_pipeline import EnhancedRAGPipeline
from ..nlp.nlp_analyzer import NLPAnalyzer
from ..analytics.analytics_tracker import AnalyticsTracker
from ..ingestion.document_processor import DocumentProcessor
from ..utils.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise AI Assistant API",
    description="Multi-Modal RAG-Powered Knowledge Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
pipeline: Optional[EnhancedRAGPipeline] = None
nlp_analyzer: Optional[NLPAnalyzer] = None
analytics_tracker: Optional[AnalyticsTracker] = None
document_processor: Optional[DocumentProcessor] = None


# ==================== Request/Response Models ====================

class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., description="User question")
    retrieval_method: str = Field("multi_query", description="Retrieval method")
    k: int = Field(5, description="Number of documents to retrieve")
    optimize_for: str = Field("balanced", description="Optimization strategy")
    include_sources: bool = Field(True, description="Include source documents")


class QueryResponse(BaseModel):
    """Query response model"""
    question: str
    answer: str
    llm_provider: str
    num_sources: int
    response_time: float
    estimated_cost: float
    sources: Optional[List[Dict]] = None
    timestamp: str


class NERRequest(BaseModel):
    """Named Entity Recognition request"""
    text: str = Field(..., description="Text to analyze")


class SentimentRequest(BaseModel):
    """Sentiment analysis request"""
    text: str = Field(..., description="Text to analyze")


class ClassificationRequest(BaseModel):
    """Text classification request"""
    text: str = Field(..., description="Text to classify")
    categories: List[str] = Field(..., description="Possible categories")


class SummarizationRequest(BaseModel):
    """Text summarization request"""
    text: str = Field(..., description="Text to summarize")
    max_length: int = Field(150, description="Maximum summary length")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global pipeline, nlp_analyzer, analytics_tracker, document_processor
    
    logger.info("Starting Enterprise AI Assistant API...")
    
    try:
        # Initialize embeddings generator
        embeddings_gen = EmbeddingsGenerator(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        
        # Load vector store
        vector_store_path = settings.EMBEDDINGS_DIR / "vector_store"
        if vector_store_path.exists():
            vector_store = FAISSVectorStore.load(str(vector_store_path))
            logger.info(f"Loaded vector store with {vector_store.index.ntotal} vectors")
        else:
            logger.warning("Vector store not found - creating empty store")
            vector_store = FAISSVectorStore(dimension=1536)
        
        # Initialize retriever
        retriever = AdvancedRAGRetriever(
            vector_store=vector_store,
            embeddings_generator=embeddings_gen,
            llm_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize LLM manager
        llm_manager = MultiLLMManager(
            openai_api_key=settings.OPENAI_API_KEY,
            default_provider="openai_gpt4_turbo"
        )
        
        # Initialize pipeline
        pipeline = EnhancedRAGPipeline(
            retriever=retriever,
            llm_manager=llm_manager,
            default_optimization="balanced"
        )
        
        # Initialize NLP analyzer
        nlp_analyzer = NLPAnalyzer(openai_api_key=settings.OPENAI_API_KEY)
        
        # Initialize analytics tracker
        analytics_tracker = AnalyticsTracker(save_dir=str(settings.DATA_DIR / "analytics"))
        
        # Initialize document processor
        # document_processor = DocumentProcessor(
        #     processed_dir=str(settings.PROCESSED_DATA_DIR),
        #     chunk_size=settings.CHUNK_SIZE,
        #     chunk_overlap=settings.CHUNK_OVERLAP
        # )
        document_processor = None

        logger.success("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Enterprise AI Assistant API...")
    
    # Save analytics
    if analytics_tracker:
        try:
            analytics_tracker.save_analytics("final_analytics.json")
            logger.info("Analytics saved")
        except Exception as e:
            logger.error(f"Failed to save analytics: {e}")


# ==================== API Endpoints ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "pipeline": "ready" if pipeline else "not initialized",
            "nlp": "ready" if nlp_analyzer else "not initialized",
            "analytics": "ready" if analytics_tracker else "not initialized"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return await root()


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Query the knowledge base
    
    - **question**: User question
    - **retrieval_method**: simple, multi_query, self_rag, hybrid
    - **k**: Number of documents to retrieve
    - **optimize_for**: cost, quality, speed, balanced
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Process query
        response = pipeline.query(
            question=request.question,
            retrieval_method=request.retrieval_method,
            k=request.k,
            optimize_for=request.optimize_for,
            include_sources=request.include_sources
        )
        
        # Track analytics in background
        if analytics_tracker:
            background_tasks.add_task(
                analytics_tracker.track_query,
                request.question,
                response
            )
        
        return QueryResponse(**response)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/ner")
async def named_entity_recognition(request: NERRequest):
    """Extract named entities from text"""
    if not nlp_analyzer:
        raise HTTPException(status_code=503, detail="NLP analyzer not initialized")
    
    try:
        entities = nlp_analyzer.named_entity_recognition(request.text)
        return {"entities": entities}
    except Exception as e:
        logger.error(f"NER failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/sentiment")
async def sentiment_analysis(request: SentimentRequest):
    """Analyze sentiment of text"""
    if not nlp_analyzer:
        raise HTTPException(status_code=503, detail="NLP analyzer not initialized")
    
    try:
        sentiment = nlp_analyzer.sentiment_analysis(request.text)
        return {"sentiment": sentiment}
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/classify")
async def text_classification(request: ClassificationRequest):
    """Classify text into categories"""
    if not nlp_analyzer:
        raise HTTPException(status_code=503, detail="NLP analyzer not initialized")
    
    try:
        classification = nlp_analyzer.text_classification(
            request.text,
            request.categories
        )
        return {"classification": classification}
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/summarize")
async def text_summarization(request: SummarizationRequest):
    """Summarize text"""
    if not nlp_analyzer:
        raise HTTPException(status_code=503, detail="NLP analyzer not initialized")
    
    try:
        summary = nlp_analyzer.summarize_text(
            request.text,
            request.max_length
        )
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a document"""
    if not document_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        # Save uploaded file
        file_path = settings.RAW_DATA_DIR / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process in background
        if background_tasks:
            background_tasks.add_task(process_document_background, str(file_path))
        
        return {
            "filename": file.filename,
            "status": "uploaded",
            "message": "Document will be processed in background"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary"""
    if not analytics_tracker:
        raise HTTPException(status_code=503, detail="Analytics tracker not initialized")
    
    try:
        summary = {
            "query_analytics": analytics_tracker.get_query_analytics(days=7),
            "cost_report": analytics_tracker.get_cost_report(days=7),
            "popular_queries": analytics_tracker.get_popular_queries(top_n=5)
        }
        return summary
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/cost")
async def get_cost_report(days: int = 30):
    """Get detailed cost report"""
    if not analytics_tracker:
        raise HTTPException(status_code=503, detail="Analytics tracker not initialized")
    
    try:
        report = analytics_tracker.get_cost_report(days=days)
        return report
    except Exception as e:
        logger.error(f"Cost report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Background Tasks ====================

async def process_document_background(file_path: str):
    """Process document in background"""
    try:
        logger.info(f"Processing document: {file_path}")
        
        # Process document
        chunks = document_processor.process_file(file_path)
        
        # Generate embeddings
        # TODO: Add to vector store
        
        logger.success(f"Processed document: {file_path} ({len(chunks)} chunks)")
        
    except Exception as e:
        logger.error(f"Background processing failed: {e}")


# ==================== Run Server ====================

if __name__ == "__main__":
    uvicorn.run(
        "src.api.api_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
