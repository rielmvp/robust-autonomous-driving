# gemini_api.py
from PIL import Image
from google import genai
from google.genai import types
import threading

thread_local = threading.local()

def get_client(api_key):
    """Get or create Gemini client (thread-local)"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = genai.Client(api_key=api_key)
    return thread_local.client

def make_api_call(api_key, model, prompt, image_paths, temperature=1, prefix=None, max_tokens=None):
    """Make Gemini API call with multiple images and text"""
    try:
        client = get_client(api_key)
        
        # Handle both single image path (string) and multiple image paths (list)
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        if prefix:
            prompt = f"{prefix}\n{prompt}"
        
        # Load all images in order
        images = []
        for i, image_path in enumerate(image_paths):
            try:
                image = Image.open(image_path)
                images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                continue
        
        if not images:
            raise ValueError("No valid images could be loaded")
        
        # Build contents array with all images first (in order), then the prompt
        contents = images + [prompt]
        
        # Single attempt with timeout (no retries for concurrency)
        config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        return response
        
    except Exception as e:
        if "No valid images could be loaded" in str(e):
            raise
        else:
            raise Exception(f"Gemini API error: {e}")

def process_response(response):
    """Process Gemini API response"""
    response_dict = {
        "text": response.text,
        "candidates": [
            {
                "content": {
                    "parts": [{"text": response.text}]
                },
                "finish_reason": getattr(response, "finish_reason", "STOP"),
                "safety_ratings": getattr(response, "safety_ratings", [])
            }
        ]
    }
    return response_dict