"""
AWS Lambda Handler
Serverless deployment for Enterprise AI Assistant
"""

import json
import os
from typing import Dict, Any
from mangum import Mangum
from .api_app import app

# Create Lambda handler using Mangum
handler = Mangum(app, lifespan="off")

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entry point
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    return handler(event, context)
