"""
AWS S3 Utility Module
Handle document upload and download from S3
"""

import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from typing import List, Optional
from loguru import logger
import json
from datetime import datetime

class S3Handler:
    """Handle AWS S3 operations for document storage"""
    
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        """
        Initialize S3 handler
        
        Args:
            bucket_name: Name of S3 bucket
            region: AWS region
        """
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=region)
        logger.info(f"S3Handler initialized for bucket: {bucket_name}")
    
    def upload_file(self, file_path: str, s3_key: Optional[str] = None) -> bool:
        """
        Upload file to S3
        
        Args:
            file_path: Local file path
            s3_key: S3 object key (defaults to filename)
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Use filename as key if not provided
        if s3_key is None:
            s3_key = f"documents/{file_path.name}"
        
        try:
            self.s3_client.upload_file(
                str(file_path),
                self.bucket_name,
                s3_key
            )
            logger.info(f"Uploaded {file_path.name} to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload {file_path.name}: {e}")
            return False
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_path
            )
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """
        List files in S3 bucket
        
        Args:
            prefix: Filter by prefix
            
        Returns:
            List of S3 object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            files = [obj['Key'] for obj in response['Contents']]
            logger.info(f"Found {len(files)} files with prefix '{prefix}'")
            return files
            
        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete {s3_key}: {e}")
            return False
    
    def upload_processed_data(self, processed_data: dict, s3_key: str) -> bool:
        """
        Upload processed document data as JSON
        
        Args:
            processed_data: Dictionary with processed document info
            s3_key: S3 object key for JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to JSON
            json_data = json.dumps(processed_data, indent=2, default=str)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            
            logger.info(f"Uploaded processed data to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload processed data: {e}")
            return False
    
    def download_processed_data(self, s3_key: str) -> Optional[dict]:
        """
        Download processed document data from S3
        
        Args:
            s3_key: S3 object key for JSON file
            
        Returns:
            Dictionary with processed data or None if failed
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            json_data = response['Body'].read().decode('utf-8')
            processed_data = json.loads(json_data)
            
            logger.info(f"Downloaded processed data from s3://{self.bucket_name}/{s3_key}")
            return processed_data
            
        except ClientError as e:
            logger.error(f"Failed to download processed data: {e}")
            return None
    
    def batch_upload(self, file_paths: List[str], prefix: str = "documents/") -> dict:
        """
        Upload multiple files to S3
        
        Args:
            file_paths: List of local file paths
            prefix: S3 key prefix
            
        Returns:
            Dictionary with upload results
        """
        results = {
            'successful': [],
            'failed': [],
            'total': len(file_paths)
        }
        
        for file_path in file_paths:
            file_path = Path(file_path)
            s3_key = f"{prefix}{file_path.name}"
            
            if self.upload_file(str(file_path), s3_key):
                results['successful'].append(str(file_path))
            else:
                results['failed'].append(str(file_path))
        
        logger.info(f"Batch upload complete: {len(results['successful'])}/{results['total']} successful")
        return results
    
    def get_file_metadata(self, s3_key: str) -> Optional[dict]:
        """
        Get metadata for S3 object
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Dictionary with metadata or None if failed
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            metadata = {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown'),
                'etag': response['ETag']
            }
            
            return metadata
            
        except ClientError as e:
            logger.error(f"Failed to get metadata for {s3_key}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    from src.utils.config import settings
    
    # Configure logging
    logger.add("s3_operations.log", rotation="1 MB")
    
    # Initialize S3 handler
    s3_handler = S3Handler(
        bucket_name=settings.AWS_S3_BUCKET,
        region=settings.AWS_REGION
    )
    
    # Example operations
    # s3_handler.upload_file("path/to/document.pdf", "documents/test.pdf")
    # files = s3_handler.list_files(prefix="documents/")
    # print(f"Found {len(files)} files")
