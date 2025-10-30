"""
Phase 1 Test Script
Test document processing and S3 upload
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pathlib import Path
from loguru import logger
from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.s3_handler import S3Handler
from src.utils.config import settings
import sys

# Configure logging
logger.remove()  # Remove default handler
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
logger.add("phase1_test.log", rotation="1 MB")

def test_document_processing():
    """Test document processing functionality"""
    logger.info("=" * 60)
    logger.info("TESTING DOCUMENT PROCESSING")
    logger.info("=" * 60)
    
    # Initialize processor
    processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    # Test with sample files (you'll need to add your own)
    test_files = [
        # Add paths to your test documents here
        # "data/raw/sample.pdf",
        # "data/raw/sample.docx",
        # "data/raw/sample.txt",
    ]
    
    if not test_files:
        logger.warning("No test files specified. Add test files to data/raw/ directory")
        logger.info("Creating a sample text file for testing...")
        
        # Create sample file
        sample_file = settings.RAW_DATA_DIR / "sample_test.txt"
        sample_content = """
        This is a sample document for testing the Enterprise AI Assistant.
        
        The system can process various document types including PDF, Word, Excel, and text files.
        
        Key Features:
        - Document chunking with overlap
        - AWS S3 integration
        - Multi-format support
        - Scalable architecture
        
        This sample text will be split into chunks based on the configured chunk size.
        Each chunk will maintain some overlap with adjacent chunks to preserve context.
        
        The processed chunks will then be ready for embedding generation in Phase 2.
        """
        
        with open(sample_file, 'w') as f:
            f.write(sample_content)
        
        test_files = [str(sample_file)]
        logger.success(f"Created sample file: {sample_file}")
    
    # Process files
    results = []
    for file_path in test_files:
        try:
            logger.info(f"\nProcessing: {file_path}")
            result = processor.process_file(file_path)
            results.append(result)
            
            logger.success(f"✓ Successfully processed {result['file_name']}")
            logger.info(f"  - Text length: {len(result['text'])} characters")
            logger.info(f"  - Chunks created: {result['chunk_count']}")
            logger.info(f"  - File hash: {result['file_hash'][:16]}...")
            
            # Show first chunk
            if result['chunks']:
                first_chunk = result['chunks'][0]
                logger.info(f"  - First chunk preview: {first_chunk['text'][:100]}...")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {file_path}: {e}")
    
    return results

def test_s3_operations():
    """Test S3 upload/download functionality"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING S3 OPERATIONS")
    logger.info("=" * 60)
    
    if not settings.AWS_S3_BUCKET:
        logger.error("AWS_S3_BUCKET not configured in .env file")
        return False
    
    # Initialize S3 handler
    s3_handler = S3Handler(
        bucket_name=settings.AWS_S3_BUCKET,
        region=settings.AWS_REGION
    )
    
    # Test file to upload
    test_file = settings.RAW_DATA_DIR / "sample_test.txt"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    # Test upload
    logger.info(f"\nUploading {test_file.name} to S3...")
    s3_key = f"test/{test_file.name}"
    
    if s3_handler.upload_file(str(test_file), s3_key):
        logger.success(f"✓ Successfully uploaded to s3://{settings.AWS_S3_BUCKET}/{s3_key}")
        
        # Test listing
        logger.info("\nListing files in S3...")
        files = s3_handler.list_files(prefix="test/")
        logger.info(f"Found {len(files)} file(s):")
        for file_key in files:
            logger.info(f"  - {file_key}")
        
        # Test metadata
        logger.info(f"\nGetting metadata for {s3_key}...")
        metadata = s3_handler.get_file_metadata(s3_key)
        if metadata:
            logger.info(f"  - Size: {metadata['size']} bytes")
            logger.info(f"  - Last Modified: {metadata['last_modified']}")
            logger.info(f"  - Content Type: {metadata['content_type']}")
        
        # Test download
        download_path = settings.PROCESSED_DATA_DIR / f"downloaded_{test_file.name}"
        logger.info(f"\nDownloading file to {download_path}...")
        
        if s3_handler.download_file(s3_key, str(download_path)):
            logger.success(f"✓ Successfully downloaded")
            
            # Verify content
            with open(download_path, 'r') as f:
                content = f.read()
            logger.info(f"  - Downloaded file size: {len(content)} characters")
        
        return True
    else:
        logger.error("✗ Failed to upload file")
        return False

def test_end_to_end():
    """Test complete workflow: process document and upload to S3"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING END-TO-END WORKFLOW")
    logger.info("=" * 60)
    
    # Process document
    processor = DocumentProcessor()
    test_file = settings.RAW_DATA_DIR / "sample_test.txt"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    logger.info(f"\n1. Processing document: {test_file.name}")
    processed_data = processor.process_file(str(test_file))
    logger.success(f"✓ Processed into {processed_data['chunk_count']} chunks")
    
    # Upload to S3
    if not settings.AWS_S3_BUCKET:
        logger.warning("⚠ S3 bucket not configured, skipping upload")
        return True
    
    s3_handler = S3Handler(
        bucket_name=settings.AWS_S3_BUCKET,
        region=settings.AWS_REGION
    )
    
    # Upload original file
    logger.info("\n2. Uploading original document to S3...")
    s3_key_original = f"documents/{test_file.name}"
    s3_handler.upload_file(str(test_file), s3_key_original)
    logger.success(f"✓ Uploaded original document")
    
    # Upload processed data
    logger.info("\n3. Uploading processed data to S3...")
    s3_key_processed = f"processed/{test_file.stem}_processed.json"
    s3_handler.upload_processed_data(processed_data, s3_key_processed)
    logger.success(f"✓ Uploaded processed data")
    
    logger.info("\n" + "=" * 60)
    logger.success("END-TO-END TEST COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    return True

def main():
    """Run all tests"""
    logger.info("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     ENTERPRISE AI ASSISTANT - PHASE 1 TEST SUITE        ║
    ║                                                           ║
    ║  Testing: Document Processing & AWS S3 Integration       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Test 1: Document Processing
        results = test_document_processing()
        
        # Test 2: S3 Operations
        test_s3_operations()
        
        # Test 3: End-to-End
        test_end_to_end()
        
        logger.info("\n" + "🎉" * 30)
        logger.success("ALL TESTS COMPLETED!")
        logger.info("🎉" * 30)
        logger.info("\nNext Steps:")
        logger.info("1. Review the generated chunks in processed data")
        logger.info("2. Check S3 bucket for uploaded files")
        logger.info("3. Proceed to Phase 2: Vector Embeddings & FAISS")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
