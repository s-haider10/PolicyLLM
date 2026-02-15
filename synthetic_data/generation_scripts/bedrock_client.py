"""AWS Bedrock Claude client for synthetic data generation."""

import json
import os
from typing import Optional


class BedrockClient:
    """Client for Claude via AWS Bedrock using native converse API."""
    
    def __init__(self, model_name: str = "claude-opus-4-5-20251101"):
        """Initialize Bedrock client.
        
        Args:
            model_name: Model to use (claude-opus-4-5-20251101, claude-sonnet, etc.)
        """
        self.model_name = model_name
        self.region = os.getenv("AWS_REGION", "us-east-2")
        
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required. Install with: pip install boto3")
        
        # Initialize boto3 Bedrock client
        # AWS credentials will come from environment variables
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
        
        # Map model names to ARNs
        self.model_arn = self._get_model_arn(model_name)
        
        print(f"✓ Bedrock client initialized with model: {self.model_name}")
        print(f"✓ Model ARN: {self.model_arn}")
        print(f"✓ AWS Region: {self.region}")
    
    def _get_model_arn(self, model_name: str) -> str:
        """Map model names to Bedrock ARNs."""
        account_id = "660201002087"
        region = self.region
        
        model_map = {
            "claude-opus-4-5-20251101": f"arn:aws:bedrock:{region}:{account_id}:inference-profile/global.anthropic.claude-opus-4-5-20251101-v1:0",
            "claude-sonnet": f"arn:aws:bedrock:{region}:{account_id}:inference-profile/global.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-haiku": f"arn:aws:bedrock:{region}:{account_id}:inference-profile/global.anthropic.claude-3-5-haiku-20241022-v1:0",
        }
        
        return model_map.get(model_name, f"arn:aws:bedrock:{region}:{account_id}:inference-profile/global.anthropic.{model_name}-v1:0")
    
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024) -> str:
        """Generate text using Claude via AWS Bedrock.
        
        Args:
            prompt: The prompt to send to Claude
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
            
            # Build Bedrock converse API call
            converse_args = {
                "modelId": self.model_arn,
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": 0.3,
                    "stopSequences": ["\n\nHuman:"]
                },
                "performanceConfig": {
                    "latency": "standard"
                }
            }
            
            # Only add system if provided
            if system:
                converse_args["system"] = [{"text": system}]
            
            # Call Bedrock converse API
            response = self.client.converse(**converse_args)
            
            # Extract text from response
            text = ""
            if "output" in response and "message" in response["output"]:
                content = response["output"]["message"].get("content", [])
                if content and len(content) > 0:
                    text = content[0].get("text", "")
            
            # Take first line to avoid repetition
            lines = text.strip().split("\n")
            if lines:
                return lines[0].strip()
            return text
            
        except Exception as e:
            print(f"Error calling Bedrock API: {e}")
            raise


