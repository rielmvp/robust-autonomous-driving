# openai_api.py
import base64
import threading
import time
from openai import OpenAI

thread_local = threading.local()

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")

def get_client(api_key, base_url=None):
    """Get or create OpenAI client (thread-local)"""
    if not hasattr(thread_local, 'client'):
        # base_url is None for the OpenAI API; set it for OpenAI-compatible servers such as vLLM
        thread_local.client = OpenAI(api_key=api_key, base_url=base_url)
    return thread_local.client

def make_api_call(api_key, model, prompt, image_paths, temperature=1, prefix=None, max_tokens=None, base_url=None):
    """Make OpenAI API call with multiple images and text"""
    client = get_client(api_key, base_url)
    
    # Handle both single image path (string) and multiple image paths (list)
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # Apply prefix if provided
    if prefix:
        prompt = f"{prefix}\n{prompt}"
    
    # Build content array with text prompt and all images
    content = [{"type": "text", "text": prompt}]
    
    # Add images in order
    for i, image_path in enumerate(image_paths):
        try:
            base64_image = encode_image_to_base64(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            continue
    
    # Reasoning models (o3/o4) don't support temperature and take max_completion_tokens instead of max_tokens
    is_reasoning_model = 'o3' in model.lower() or 'o4' in model.lower()
    use_temperature = not is_reasoning_model
    
    # Retry logic with exponential backoff
    for attempt in range(3):
        try:
            # Build API call parameters
            api_params = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }
            
            # Only add temperature if the model supports it
            if use_temperature:
                api_params["temperature"] = temperature
            
            if max_tokens:
                api_params["max_completion_tokens" if is_reasoning_model else "max_tokens"] = max_tokens
            
            response = client.chat.completions.create(**api_params)
            return response.model_dump()
            
        except Exception as e:
            print(f"OpenAI API attempt {attempt+1} failed: {e}")
            if attempt < 2:  # Only retry if we haven't exceeded max attempts
                wait_time = (2 ** attempt) * 30  # Exponential backoff: 30s, 60s
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

def process_response(response):
    """Process OpenAI API response"""
    return response  # OpenAI response is already in the right format