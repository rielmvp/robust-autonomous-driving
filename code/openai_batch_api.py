# openai_batch_api.py
import argparse
import json
import os
import base64
import math
import time
import datetime
import concurrent.futures
from pathlib import Path
from openai import OpenAI



def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")

def get_image_paths_from_item(item):
    """Get list of image paths from item data"""
    # Handle both old format (single image_path) and new format (image_folder + image_filenames)
    if 'image_path' in item:
        # Old format - single image
        return [item['image_path']]
    elif 'image_folder' in item and 'image_filenames' in item:
        # New format - multiple images
        image_folder = item['image_folder']
        image_filenames = item['image_filenames']
        # Join image folder with each filename, maintaining order
        image_paths = []
        for filename in image_filenames:
            full_path = os.path.join(image_folder, filename)
            image_paths.append(full_path)
        return image_paths
    else:
        raise ValueError(f"Item {item.get('ID', 'unknown')} missing image path information")

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)

def estimate_request_size(item, image_paths):
    """Estimate the size of a single request in bytes"""
    # Base size of JSON structure
    base_size = 500  # Approximate size of JSON structure in bytes
    
    # Add size of prompt text
    prompt_size = len(item['prompt'].encode('utf-8'))
    
    # Add size of base64-encoded images (file size * 4/3 for base64 overhead)
    image_size = 0
    for image_path in image_paths:
        if os.path.exists(image_path):
            image_size += os.path.getsize(image_path) * 4/3  # Base64 encoding overhead
    
    return base_size + prompt_size + image_size

def create_size_based_batches(data, max_batch_size_mb):
    """Create batches based on maximum batch size in MB"""
    max_batch_size_bytes = max_batch_size_mb * 1024 * 1024
    batches = []
    current_batch = []
    current_batch_size = 0
    
    for item in data:
        try:
            # Estimate size of this item
            image_paths = get_image_paths_from_item(item)
            item_size = estimate_request_size(item, image_paths)
            
            # If this single item is larger than max batch size, log warning and include it anyway
            if item_size > max_batch_size_bytes:
                print(f"Warning: Item {item['ID']} exceeds max batch size ({item_size/(1024*1024):.2f}MB > {max_batch_size_mb}MB)")
                # Start a new batch for this large item
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_size = 0
                
                # Add this large item as its own batch
                batches.append([item])
                continue
            
            # If adding this item would exceed max size, start a new batch
            if current_batch and current_batch_size + item_size > max_batch_size_bytes:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
            
            # Add item to current batch
            current_batch.append(item)
            current_batch_size += item_size
            
        except Exception as e:
            print(f"Error estimating size for item {item['ID']}: {e}")
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches

def split_data_into_batches(data, num_batches):
    """Split data into equal-sized batches"""
    if num_batches <= 0:
        num_batches = 1
    
    # Calculate batch size (ensure at least 1 item per batch)
    batch_size = max(1, math.ceil(len(data) / num_batches))
    
    # Split data into batches
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    
    # If we have more batches than requested (due to rounding),
    # combine the last batches
    while len(batches) > num_batches:
        last_batch = batches.pop()
        batches[-1].extend(last_batch)
    
    return batches

def save_batch_info(batch_info):
    """Save batch info to file with current timestamp"""
    batch_info_file = os.path.join(batch_info['batch_dir'], "batch_info.json")
    batch_info['last_updated'] = datetime.datetime.now().isoformat()
    with open(batch_info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)

def prepare_batch(data, model, temperature, task_name, batch_size_mb=185, num_batches=None, prefix=None, test_mode=False, max_tokens=None):
    """Prepare batch files for OpenAI batch API"""
    if test_mode:
        data = data[:2]
        batch_size_mb = min(1, batch_size_mb)
    
    # Create batch directory structure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.replace('-', '_')
    batch_dir = os.path.join("batches", task_name, f"{model_name}_{timestamp}")
    ensure_dir_exists(batch_dir)
    
    # Split data into batches
    if batch_size_mb is not None:
        print(f"Splitting data into batches with maximum size of {batch_size_mb}MB")
        data_batches = create_size_based_batches(data, batch_size_mb)
        print(f"Created {len(data_batches)} batches based on size constraint")
    else:
        num_batches = num_batches or 9
        print(f"Splitting data into {num_batches} equal-sized batches")
        data_batches = split_data_into_batches(data, num_batches)
    
    # Initialize batch states tracking
    batch_states = {}
    
    # Create batch files
    batch_files = []
    for batch_idx, batch_data in enumerate(data_batches):
        print(f"Preparing batch {batch_idx+1}/{len(data_batches)} with {len(batch_data)} items")
        
        batch_requests = []
        for item in batch_data:
            try:
                # Get image paths (can be single or multiple images)
                image_paths = get_image_paths_from_item(item)
                
                # Apply prefix to prompt if specified
                prompt_text = item['prompt']
                if prefix:
                    prompt_text = f"{prefix}\n{prompt_text}"
                
                # Build content array with text prompt and all images
                content = [{"type": "text", "text": prompt_text}]
                
                # Add images in order
                for image_path in image_paths:
                    base64_image = encode_image_to_base64(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                
                # Check if model supports temperature parameter
                supports_temperature = not (model.startswith('o3') or model.startswith('o4-mini'))
                
                # Build request body
                request_body = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                }
                
                # Only add temperature if the model supports it
                if supports_temperature:
                    request_body["temperature"] = temperature
                
                # Reasoning models take max_completion_tokens instead of max_tokens
                if max_tokens:
                    request_body["max_tokens" if supports_temperature else "max_completion_tokens"] = max_tokens
                
                request = {
                    "custom_id": item['ID'],
                    "method": "POST", 
                    "url": "/v1/chat/completions",
                    "body": request_body
                }
                
                batch_requests.append(request)
                
            except Exception as e:
                print(f"Error preparing item {item['ID']}: {e}")
        
        # Save the batch as a JSONL file
        batch_file = os.path.join(batch_dir, f"batch_{batch_idx+1}_of_{len(data_batches)}.jsonl")
        
        with open(batch_file, 'w') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
        
        batch_size_mb_actual = os.path.getsize(batch_file) / (1024 * 1024)
        print(f"Created batch file {batch_file} with {len(batch_requests)} requests ({batch_size_mb_actual:.2f} MB)")
        batch_files.append(batch_file)
        
        # Initialize batch state
        batch_states[f"batch_{batch_idx+1}"] = {
            "batch_file": batch_file,
            "status": "prepared",
            "file_id": None,
            "batch_id": None,
            "created_at": None,
            "completed_at": None,
            "results_file": None,
            "processed_file": None,
            "error_message": None
        }
    
    # Save batch information
    batch_info = {
        "batch_dir": batch_dir,
        "batch_files": batch_files,
        "model": model,
        "temperature": temperature,
        "total_requests": len(data),
        "num_batches": len(data_batches),
        "max_batch_size_mb": batch_size_mb,
        "test": test_mode,
        "prefix": prefix,
        "created_at": datetime.datetime.now().isoformat(),
        "batch_states": batch_states
    }
    
    save_batch_info(batch_info)
    
    print(f"Batch preparation complete. Created {len(batch_files)} batch files in {batch_dir}")
    return batch_info

def upload_batch_file(client, batch_file):
    """Upload batch file to OpenAI"""
    for attempt in range(3):
        try:
            print(f"Uploading batch file: {batch_file}")
            with open(batch_file, 'rb') as f:
                file_obj = client.files.create(
                    file=f,
                    purpose="batch"
                )
            return file_obj
        except Exception as e:
            print(f"Attempt {attempt+1} failed to upload batch file: {e}")
            if attempt < 2:  # Only sleep if we're going to retry
                print(f"Retrying in 60 seconds...")
                time.sleep(60)
            else:
                raise

def create_batch_job(client, file_id):
    """Create batch job with OpenAI"""
    for attempt in range(3):
        try:
            print(f"Creating batch with file ID: {file_id}")
            batch = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            return batch
        except Exception as e:
            print(f"Attempt {attempt+1} failed to create batch: {e}")
            if attempt < 2:  # Only sleep if we're going to retry
                print(f"Retrying in 60 seconds...")
                time.sleep(60)
            else:
                raise

def check_batch_status(client, batch_id):
    """Check batch status"""
    for attempt in range(3):
        try:
            batch = client.batches.retrieve(batch_id)
            return batch
        except Exception as e:
            print(f"Attempt {attempt+1} failed to check batch status: {e}")
            if attempt < 2:  # Only sleep if we're going to retry
                print(f"Retrying in 60 seconds...")
                time.sleep(60)
            else:
                raise

def download_batch_results(client, file_id, output_file):
    """Download batch results"""
    for attempt in range(3):
        try:
            print(f"Downloading results from file ID: {file_id}")
            response = client.files.content(file_id)
            
            # Ensure directory exists
            ensure_dir_exists(os.path.dirname(output_file))
            
            # Save the raw response
            with open(output_file, 'w') as f:
                f.write(response.text)
                
            return response.text
        except Exception as e:
            print(f"Attempt {attempt+1} failed to download results: {e}")
            if attempt < 2:  # Only sleep if we're going to retry
                print(f"Retrying in 60 seconds...")
                time.sleep(60)
            else:
                raise

def process_batch_results(results_text, original_data):
    """Process batch results and combine with original data"""
    # Parse the JSONL results
    results = []
    for line in results_text.strip().split('\n'):
        if line.strip():
            results.append(json.loads(line))
    
    # Create a mapping from custom_id to result
    results_map = {r['custom_id']: r for r in results}
    
    # Combine original data with results
    combined_results = []
    for item in original_data:
        item_id = item['ID']
        if item_id in results_map:
            item_with_response = item.copy()
            item_with_response['api_response'] = results_map[item_id]
            combined_results.append(item_with_response)
    
    return combined_results

def load_batch_state(batch_info):
    """Load the current batch state from disk"""
    batch_info_file = os.path.join(batch_info['batch_dir'], "batch_info.json")
    if os.path.exists(batch_info_file):
        try:
            with open(batch_info_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not load batch state from {batch_info_file}")
    return batch_info

def find_incomplete_batches(batch_info):
    """Find batches that are not yet completed"""
    batch_states = batch_info.get('batch_states', {})
    incomplete_batches = []
    
    for batch_key, state in batch_states.items():
        if state.get('status') not in ['completed', 'failed']:
            incomplete_batches.append((batch_key, state))
    
    return incomplete_batches

def submit_and_process_batches(api_key, batch_info, output_file, check_interval=10, concurrent_batches=10):
    """Submit batches and process results with resume support"""
    client = OpenAI(api_key=api_key)
    
    # Load current batch state (might have been updated since initial creation)
    batch_info = load_batch_state(batch_info)
    
    # Load original data
    with open(batch_info.get('input_file', 'dataset/ground_truth_generation.json'), 'r') as f:
        original_data = json.load(f)
    
    results_dir = os.path.dirname(output_file)
    ensure_dir_exists(results_dir)
    
    # Check for incomplete batches
    incomplete_batches = find_incomplete_batches(batch_info)
    completed_batches = []
    
    # Load already completed results
    all_results = []
    for batch_key, state in batch_info.get('batch_states', {}).items():
        if state.get('status') == 'completed' and state.get('processed_file'):
            processed_file = state['processed_file']
            if os.path.exists(processed_file):
                try:
                    with open(processed_file, 'r') as f:
                        batch_results = json.load(f)
                    all_results.extend(batch_results)
                    completed_batches.append(batch_key)
                    print(f"✓ Loaded {len(batch_results)} results from completed batch {batch_key}")
                except json.JSONDecodeError:
                    print(f"Warning: Could not load results from {processed_file}")
                    # Mark this batch as needing reprocessing
                    incomplete_batches.append((batch_key, state))
    
    if completed_batches:
        print(f"Found {len(completed_batches)} already completed batches with {len(all_results)} total results")
    
    if not incomplete_batches:
        print("All batches are already completed!")
        # Save final results
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        return all_results
    
    print(f"Processing {len(incomplete_batches)} incomplete batches with {concurrent_batches} workers")
    
    def process_single_batch(batch_key_and_state):
        """Process a single batch and return the results"""
        batch_key, batch_state = batch_key_and_state
        batch_file = batch_state['batch_file']
        batch_name = os.path.basename(batch_file).replace(".jsonl", "")
        
        print(f"Processing batch {batch_key}: {batch_name}")
        
        # Check if this batch already has file_id and batch_id (resume case)
        file_id = batch_state.get('file_id')
        batch_id = batch_state.get('batch_id')
        
        try:
            # Upload the batch file if not already uploaded
            if not file_id:
                file_obj = upload_batch_file(client, batch_file)
                file_id = file_obj.id
                print(f"Uploaded file with ID: {file_id}")
                
                # Update batch state immediately
                batch_info['batch_states'][batch_key]['file_id'] = file_id
                batch_info['batch_states'][batch_key]['status'] = 'uploaded'
                save_batch_info(batch_info)
            else:
                print(f"Resuming with existing file ID: {file_id}")
            
            # Create a batch if not already created
            if not batch_id:
                batch = create_batch_job(client, file_id)
                batch_id = batch.id
                print(f"Created batch with ID: {batch_id}")
                
                # Update batch state immediately
                batch_info['batch_states'][batch_key]['batch_id'] = batch_id
                batch_info['batch_states'][batch_key]['status'] = 'submitted'
                batch_info['batch_states'][batch_key]['created_at'] = datetime.datetime.now().isoformat()
                save_batch_info(batch_info)
            else:
                print(f"Resuming with existing batch ID: {batch_id}")
            
            # Create batch specific directory
            this_batch_dir = os.path.join(results_dir, batch_id)
            ensure_dir_exists(this_batch_dir)
            
            # Save batch details
            batch_details = {
                "batch_id": batch_id,
                "file_id": file_id,
                "batch_file": batch_file,
                "status": batch_state.get('status', 'submitted'),
                "created_at": batch_state.get('created_at')
            }
            with open(os.path.join(this_batch_dir, "batch_details.json"), 'w') as f:
                json.dump(batch_details, f, indent=2)
            
            # Monitor the batch status
            completed = False
            while not completed:
                print(f"Checking status of batch {batch_id}...")
                batch = check_batch_status(client, batch_id)
                print(f"Batch status: {batch.status}")
                
                # Update batch state
                batch_info['batch_states'][batch_key]['status'] = batch.status
                save_batch_info(batch_info)
                
                # Update batch details with current status
                batch_details["status"] = batch.status
                batch_details["updated_at"] = time.time()
                with open(os.path.join(this_batch_dir, "batch_details.json"), 'w') as f:
                    json.dump(batch_details, f, indent=2)
                
                if batch.status == 'completed':
                    completed = True
                elif batch.status in ['failed', 'expired', 'cancelled']:
                    batch_info['batch_states'][batch_key]['status'] = 'failed'
                    batch_info['batch_states'][batch_key]['error_message'] = f"Batch failed with status: {batch.status}"
                    save_batch_info(batch_info)
                    raise Exception(f"Batch failed with status: {batch.status}")
                else:
                    print(f"Waiting {check_interval} seconds before checking again...")
                    time.sleep(check_interval)
            
            # Download the results
            batch_results = []
            if batch.output_file_id:
                raw_results_file = os.path.join(this_batch_dir, "raw_results.jsonl")
                results_text = download_batch_results(client, batch.output_file_id, raw_results_file)
                print(f"Downloaded results to {raw_results_file}")
                
                # Update batch state
                batch_info['batch_states'][batch_key]['results_file'] = raw_results_file
                
                # Process any error files if they exist
                if batch.error_file_id:
                    error_file = os.path.join(this_batch_dir, "errors.jsonl")
                    error_text = download_batch_results(client, batch.error_file_id, error_file)
                    print(f"Downloaded errors to {error_file}")
                
                # Process the batch results
                batch_results = process_batch_results(results_text, original_data)
                
                # Save this batch's processed results
                processed_file = os.path.join(this_batch_dir, "processed_results.json")
                with open(processed_file, 'w') as f:
                    json.dump(batch_results, f, indent=2)
                
                # Update batch state to completed
                batch_info['batch_states'][batch_key]['status'] = 'completed'
                batch_info['batch_states'][batch_key]['completed_at'] = datetime.datetime.now().isoformat()
                batch_info['batch_states'][batch_key]['processed_file'] = processed_file
                save_batch_info(batch_info)
                
                print(f"Saved {len(batch_results)} results from batch {batch_id}")
            
            return batch_results
        
        except Exception as e:
            print(f"Error processing batch {batch_name}: {e}")
            batch_info['batch_states'][batch_key]['status'] = 'failed'
            batch_info['batch_states'][batch_key]['error_message'] = str(e)
            save_batch_info(batch_info)
            return []
    
    # Process incomplete batches
    if concurrent_batches <= 1:
        # Process batches sequentially
        for batch_key_and_state in incomplete_batches:
            batch_results = process_single_batch(batch_key_and_state)
            all_results.extend(batch_results)
    else:
        # Process batches concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_batches) as executor:
            futures = []
            for batch_key_and_state in incomplete_batches:
                future = executor.submit(process_single_batch, batch_key_and_state)
                futures.append(future)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as exc:
                    print(f"Batch processing generated exception: {exc}")
    
    # Save combined results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"All done! Processed {len(all_results)} items across {len(batch_info.get('batch_states', {}))} batches.")
    print(f"Combined results saved to {output_file}")
    
    return all_results

def process_response(response):
    """Process OpenAI API response"""
    return response  # OpenAI response is already in the right format