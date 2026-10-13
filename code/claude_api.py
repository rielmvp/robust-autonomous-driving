# claude_api.py
import base64
import threading
from pathlib import Path
from anthropic import Anthropic

thread_local = threading.local()

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")

def get_image_media_type(image_path):
    """Get media type for image"""
    ext = Path(image_path).suffix.lower()
    if ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        print(f"Warning: Unknown image type for {image_path}, defaulting to image/jpeg.")
        return "image/jpeg"

def get_client(api_key):
    """Get or create Claude client (thread-local)"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = Anthropic(api_key=api_key)
    return thread_local.client

def make_api_call(api_key, model, prompt, image_paths, temperature=1.0, max_tokens=1024, prefix=None):
    """Make Claude API call with multiple images and text"""
    client = get_client(api_key)
    
    # Handle both single image path (string) and multiple image paths (list)
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    if prefix:
        prompt = f"{prefix}\n{prompt}"

    # Build content array with all images first (in order), then the text prompt
    content = []
    
    # Add images in order
    for i, image_path in enumerate(image_paths):
        try:
            base64_image = encode_image_to_base64(image_path)
            media_type = get_image_media_type(image_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_image,
                },
            })
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            continue
    
    # Add text prompt at the end
    content.append({"type": "text", "text": prompt})
    
    if not any(item["type"] == "image" for item in content):
        raise ValueError("No valid images could be loaded")

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            )
            return response.model_dump()
        except Exception as e:
            print(f"Claude API attempt {attempt+1} failed: {e}")
            if attempt < 2:  
                print(f"Retrying in 60 seconds...")
                import time
                time.sleep(60)
            else:
                raise

def process_response(response):
    """Process Claude API response"""
    return response  # Claude response is already in the right format