# claude_batch_api.py
import argparse
import json
import os
import base64
import re
import math
import time
import datetime
import concurrent.futures
from pathlib import Path
from anthropic import Anthropic, APIError, APIStatusError

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
    elif ext == ".gif":
        return "image/gif"
    elif ext == ".webp":
        return "image/webp"
    else:
        print(f"Warning: Unknown image type for {image_path}, defaulting to image/jpeg.")
        return "image/jpeg"

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

def sanitize_custom_id(id_value):
    """
    Sanitize ID to ensure it complies with Claude Batch API requirements:
    - Only alphanumeric characters, underscores, and hyphens
    - Between 1-64 characters in length
    """
    if not id_value:
        return f"empty_id_{datetime.datetime.now().strftime('%H%M%S')}", True, "Empty ID replaced"
    
    original_id = str(id_value)
    
    # Check if ID already meets requirements
    if re.match(r'^[a-zA-Z0-9_-]{1,64}$', original_id):
        return original_id, False, None
    
    # Remove invalid characters
    sanitized_id = re.sub(r'[^a-zA-Z0-9_-]', '', original_id)
    
    # Truncate if too long
    if len(sanitized_id) > 64:
        sanitized_id = sanitized_id[:64]
    
    # If empty after sanitizing, create a placeholder
    if not sanitized_id:
        sanitized_id = f"invalid_id_{datetime.datetime.now().strftime('%H%M%S')}"
    
    return sanitized_id, True, f"ID '{original_id}' was modified to '{sanitized_id}'"

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)

def estimate_request_size(item, image_paths):
    """Estimate the size of a single request in bytes"""
    # Base size of JSON structure
    base_size = 1000  # Approximate size of JSON structure in bytes (higher for Claude's structure)
    
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
    batch_size = max(1, math.ceil(len(data) / num_batches))
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    # Consolidate if more batches than requested due to ceil
    while len(batches) > num_batches and len(batches) > 1:
        last_batch = batches.pop()
        batches[-1].extend(last_batch)
    return batches

def save_batch_info(batch_info):
    """Save batch info to file with current timestamp"""
    batch_info_file = os.path.join(batch_info['batch_dir'], "batch_info.json")
    batch_info['last_updated'] = datetime.datetime.now().isoformat()
    with open(batch_info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)

def prepare_batch(data, model, temperature, task_name, max_tokens=2048, batch_size_mb=240, num_batches=None, prefix=None, test_mode=False):
    """Prepare batch files for Claude batch API"""
    if test_mode:
        data = data[:2]
        batch_size_mb = min(1, batch_size_mb)
    
    # Create batch directory structure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model.replace("/", "_").replace(":", "_")
    batch_dir = os.path.join("batches", task_name, f"{safe_model_name}_{timestamp}")
    ensure_dir_exists(batch_dir)
    
    # Split data into batches
    if batch_size_mb is not None:
        print(f"Splitting data into batches with maximum size of {batch_size_mb}MB")
        data_batches = create_size_based_batches(data, batch_size_mb)
        print(f"Created {len(data_batches)} batches based on size constraint")
    else:
        num_batches = num_batches or 1
        print(f"Splitting data into {num_batches} equal-sized batches")
        data_batches = split_data_into_batches(data, num_batches)
    
    # Create batch files
    batch_files = []
    modified_ids = []  # Track all modified IDs
    
    for batch_idx, batch_data in enumerate(data_batches):
        print(f"Preparing batch {batch_idx+1}/{len(data_batches)} with {len(batch_data)} items")
        
        batch_requests_payload = []  # This will be the "requests" array for Claude Batch API
        for item_idx, item in enumerate(batch_data):
            try:
                # Sanitize the ID for Claude Batch API compliance
                original_id = item['ID']
                sanitized_id, was_modified, modification_message = sanitize_custom_id(original_id)
                
                # Track modified IDs
                if was_modified:
                    print(f"🔧 Item {item_idx} in batch {batch_idx+1}: {modification_message}")
                    modified_ids.append({
                        "original_id": original_id,
                        "sanitized_id": sanitized_id,
                        "batch_index": batch_idx,
                        "item_index": item_idx
                    })
                
                # Get image paths (can be single or multiple images)
                image_paths = get_image_paths_from_item(item)

                # Apply prefix to prompt if specified
                prompt_text = item['prompt']
                if prefix:
                    prompt_text = f"{prefix}\n{prompt_text}"

                # Build content array with all images first (in order), then the text prompt
                content = []
                
                # Add images in order
                for image_path in image_paths:
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
                
                # Add text prompt at the end
                content.append({"type": "text", "text": prompt_text})

                # This is the structure for a single request within the Claude Batch API
                request_params = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]
                }
                
                claude_batch_request_item = {
                    "custom_id": sanitized_id,  # Use sanitized ID instead of original
                    "params": request_params  # The actual message creation parameters
                }
                batch_requests_payload.append(claude_batch_request_item)
                
            except Exception as e:
                print(f"❌ Error preparing item {item.get('ID', 'unknown')} at index {item_idx}: {e}")
        
        # Save this batch payload as a single JSONL file
        batch_file_path = os.path.join(batch_dir, f"batch_input_{batch_idx+1}_of_{len(data_batches)}.jsonl")
        
        with open(batch_file_path, 'w') as f:
            for req_item in batch_requests_payload:
                f.write(json.dumps(req_item) + '\n')
        
        batch_size_mb_actual = os.path.getsize(batch_file_path) / (1024 * 1024)
        print(f"Created batch input file {batch_file_path} with {len(batch_requests_payload)} requests ({batch_size_mb_actual:.2f} MB)")
        batch_files.append(batch_file_path)
    
    # Save a record of modified IDs for reference
    if modified_ids:
        modified_ids_path = os.path.join(batch_dir, "modified_ids.json")
        with open(modified_ids_path, 'w') as f:
            json.dump(modified_ids, f, indent=2)
        print(f"✅ Modified {len(modified_ids)} IDs to comply with API requirements. Details saved to {modified_ids_path}")
    
    # Initialize batch states tracking
    batch_states = {}
    for i, batch_file in enumerate(batch_files):
        batch_states[f"batch_{i+1}"] = {
            "batch_file": batch_file,
            "status": "prepared",
            "api_batch_id": None,
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
        "max_tokens": max_tokens,
        "total_requests": len(data),
        "num_batches": len(data_batches),
        "max_batch_size_mb": batch_size_mb,
        "test": test_mode,
        "has_modified_ids": len(modified_ids) > 0,
        "modified_ids_file": "modified_ids.json" if modified_ids else None,
        "prefix": prefix,
        "created_at": datetime.datetime.now().isoformat(),
        "batch_states": batch_states
    }
    
    save_batch_info(batch_info)
    
    print(f"Batch preparation complete. Created {len(batch_files)} batch files in {batch_dir}")
    return batch_info

def to_iso_format_if_datetime(dt_obj):
    """Convert datetime to ISO format if it's a datetime object"""
    if isinstance(dt_obj, datetime.datetime):
        return dt_obj.isoformat()
    return dt_obj

def create_claude_batch(client, requests_payload):
    """Create Claude batch job"""
    for attempt in range(3):
        try:
            print(f"Creating Claude batch with {len(requests_payload)} requests...")
            batch_response = client.messages.batches.create(requests=requests_payload)
            return batch_response
        except (APIError, APIStatusError) as e:
            print(f"Attempt {attempt+1} failed to create batch: {e}")
            status_code = getattr(e, 'status_code', None)
            if status_code == 429: 
                time.sleep(60)
            elif attempt < 2: 
                time.sleep(30)
            else: 
                raise
        except TypeError as te:
            print(f"TypeError during batch creation: {te}. SDK path/args for 'create' might be wrong.")
            raise
        except Exception as e:
            print(f"Attempt {attempt+1} unexpected error during batch creation: {e}")
            if attempt < 2: 
                time.sleep(30)
            else: 
                raise

def check_claude_batch_status(client, batch_id_str):
    """Check Claude batch status"""
    for attempt in range(3):
        try:
            print(f"Attempting to retrieve status for batch ID: {batch_id_str}")
            batch_status_response = client.messages.batches.retrieve(batch_id_str)
            return batch_status_response
        except (APIError, APIStatusError) as e:
            print(f"Attempt {attempt+1} failed to check status for {batch_id_str}: {e}")
            status_code = getattr(e, 'status_code', None)
            if status_code == 429: 
                time.sleep(60)
            elif attempt < 2: 
                time.sleep(30)
            else: 
                raise
        except TypeError as te:
            print(f"TypeError checking status for {batch_id_str}: {te}. SDK path/args for 'retrieve' might be wrong.")
            raise
        except Exception as e:
            print(f"Attempt {attempt+1} unexpected error checking status for {batch_id_str}: {e}")
            if attempt < 2: 
                time.sleep(30)
            else: 
                raise

def stream_and_save_claude_batch_results(client, batch_id_str, output_file_path):
    """Stream and save Claude batch results"""
    all_results_lines_as_json_strings = []
    try:
        print(f"Streaming results for batch: {batch_id_str} using client.messages.batches.results()")
        for result_sdk_object in client.messages.batches.results(batch_id_str):
            result_dict = None
            if hasattr(result_sdk_object, 'model_dump'):
                result_dict = result_sdk_object.model_dump(exclude_none=True)
            elif hasattr(result_sdk_object, 'dict'):
                result_dict = result_sdk_object.dict(exclude_none=True)
            elif hasattr(result_sdk_object, 'to_dict'):
                result_dict = result_sdk_object.to_dict()
            elif isinstance(result_sdk_object, dict):  # If it's already a dict
                result_dict = result_sdk_object
            else:
                print(f"Warning: No standard dict conversion for {type(result_sdk_object)}. Trying vars().")
                try: 
                    result_dict = vars(result_sdk_object)
                except TypeError:
                    print(f"Error: vars() failed for {result_sdk_object}. Skipping.")
                    continue
            
            if result_dict is not None:
                # Basic check for nested datetimes; ideally, this needs to be recursive for deep structures
                # but batch result lines are often flat or have limited nesting of special types.
                serializable_result_dict = {}
                for k, v in result_dict.items():
                    if isinstance(v, dict):  # simple check for one level of nesting
                        serializable_result_dict[k] = {nk: to_iso_format_if_datetime(nv) for nk, nv in v.items()}
                    elif isinstance(v, list):  # if a list contains dicts with datetimes
                        new_list = []
                        for item_in_list in v:
                            if isinstance(item_in_list, dict):
                                new_list.append({nk: to_iso_format_if_datetime(nv) for nk, nv in item_in_list.items()})
                            else:
                                new_list.append(to_iso_format_if_datetime(item_in_list))
                        serializable_result_dict[k] = new_list
                    else:
                        serializable_result_dict[k] = to_iso_format_if_datetime(v)
                all_results_lines_as_json_strings.append(json.dumps(serializable_result_dict))
            else:
                print(f"Warning: Could not convert SDK result object to dict for batch {batch_id_str}")

        if not all_results_lines_as_json_strings:
            print(f"Warning: No results successfully converted/streamed for batch {batch_id_str}.")
            return ""
        results_text_content = "\n".join(all_results_lines_as_json_strings)
        ensure_dir_exists(os.path.dirname(output_file_path))
        with open(output_file_path, 'w', encoding='utf-8') as f: 
            f.write(results_text_content)
        print(f"Streamed results saved to {output_file_path}")
        return results_text_content
    except (APIError, APIStatusError) as e:
        print(f"API error streaming results for {batch_id_str}: {e}")
        raise
    except TypeError as te:
        print(f"TypeError streaming results for {batch_id_str}: {te}. Problem with obj to dict or JSON dump.")
        raise
    except Exception as e:
        print(f"Unexpected error streaming results for {batch_id_str}: {e}")
        raise


def process_downloaded_results(results_text, original_data_map):
    """Process downloaded results and combine with original data"""
    processed_items = []
    if not results_text: 
        return processed_items
    
    for line in results_text.strip().split('\n'):
        if not line.strip(): 
            continue
        try:
            result_item = json.loads(line)
            custom_id = result_item.get('custom_id')
            
            if custom_id:
                # Try to find the original data using the custom_id
                original_item = None
                
                # First, try direct lookup
                if custom_id in original_data_map:
                    original_item = original_data_map[custom_id]
                else:
                    # If not found, try to find by searching through all items
                    # This handles cases where the ID format might be different
                    for orig_id, orig_item in original_data_map.items():
                        if orig_id == custom_id or str(orig_id) == str(custom_id):
                            original_item = orig_item
                            break
                
                if original_item:
                    combined_item = original_item.copy()
                    combined_item['api_response'] = result_item
                    processed_items.append(combined_item)
                    print(f"✓ Processed result for ID: {custom_id}")
                else:
                    print(f"⚠️ Warning: custom_id '{custom_id}' not found in original data")
                    # Create a minimal item with just the API response
                    minimal_item = {
                        'ID': custom_id,
                        'api_response': result_item
                    }
                    processed_items.append(minimal_item)
            else:
                print(f"⚠️ Warning: Result item missing custom_id")
                
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Could not decode JSON line: {line[:100]}... Error: {e}")
            continue
    
    print(f"✓ Successfully processed {len(processed_items)} results")
    return processed_items


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
    client = Anthropic(api_key=api_key, default_headers={"anthropic-version": "2023-06-01"})
    
    # Load current batch state (might have been updated since initial creation)
    batch_info = load_batch_state(batch_info)
    
    # Load original data with better error handling
    input_file = batch_info.get('input_file', 'dataset/ground_truth_generation.json')
    try:
        with open(input_file, 'r') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Input file {input_file} not found. Will process results without original data.")
        original_data = []
    
    # Create a mapping from original IDs to data (need to handle modified IDs)
    original_data_map = {}
    for item in original_data:
        if 'ID' in item:
            original_data_map[item['ID']] = item
            # Also add string version in case of type mismatch
            original_data_map[str(item['ID'])] = item
    
    # If there were modified IDs, we need to update the mapping
    if batch_info.get('has_modified_ids', False):
        modified_ids_file = os.path.join(batch_info['batch_dir'], 'modified_ids.json')
        if os.path.exists(modified_ids_file):
            try:
                with open(modified_ids_file, 'r') as f:
                    modified_ids = json.load(f)
                # Add mappings for sanitized IDs
                for mod_id in modified_ids:
                    if mod_id['original_id'] in original_data_map:
                        original_data_map[mod_id['sanitized_id']] = original_data_map[mod_id['original_id']]
                        original_data_map[str(mod_id['sanitized_id'])] = original_data_map[mod_id['original_id']]
                print(f"✓ Loaded {len(modified_ids)} modified ID mappings")
            except Exception as e:
                print(f"Warning: Could not load modified IDs: {e}")
    
    print(f"✓ Created mapping for {len(original_data_map)} original data items")
    
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
    
    print(f"Processing {len(incomplete_batches)} incomplete batches with {concurrent_batches} workers.")
    
    def process_single_claude_batch_file(batch_key_and_state):
        """Process a single Claude batch file and return the results"""
        batch_key, batch_state = batch_key_and_state
        batch_input_file_path = batch_state['batch_file']
        batch_file_name = os.path.basename(batch_input_file_path)
        
        print(f"Processing batch {batch_key}: {batch_file_name}")
        
        # Check if this batch already has an API batch ID (resume case)
        api_batch_id = batch_state.get('api_batch_id')
        
        if api_batch_id:
            print(f"Resuming existing batch {batch_key} with API ID: {api_batch_id}")
        else:
            # Load requests from file
            requests_payload = []
            with open(batch_input_file_path, 'r') as f:
                for line in f:
                    try: 
                        requests_payload.append(json.loads(line))
                    except json.JSONDecodeError as e: 
                        print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
            
            if not requests_payload:
                print(f"No valid requests in {batch_input_file_path}. Skipping.")
                # Update batch state
                batch_info['batch_states'][batch_key]['status'] = 'failed'
                batch_info['batch_states'][batch_key]['error_message'] = 'No valid requests in file'
                save_batch_info(batch_info)
                return []

        batch_run_details = {
            "api_batch_id": api_batch_id, 
            "source_input_file": batch_input_file_path,
            "processing_status": batch_state.get('status', 'pending_creation'), 
            "created_at_api": batch_state.get('created_at'),
            "total_requests_in_batch": len(requests_payload) if not api_batch_id else None, 
            "ended_at_api": batch_state.get('completed_at'),
            "results_url": None, 
            "request_counts": None,
            "updated_at_script": to_iso_format_if_datetime(datetime.datetime.now(datetime.timezone.utc))
        }
        this_batch_results_dir = results_dir

        try:
            # Create batch if not already created
            if not api_batch_id:
                batch_creation_response = create_claude_batch(client, requests_payload)
                api_batch_id = batch_creation_response.id
                batch_run_details["api_batch_id"] = api_batch_id
                print(f"Created Claude batch with ID: {api_batch_id} from {batch_file_name}")
                
                # Update batch state immediately
                batch_info['batch_states'][batch_key]['api_batch_id'] = api_batch_id
                batch_info['batch_states'][batch_key]['status'] = 'submitted'
                batch_info['batch_states'][batch_key]['created_at'] = datetime.datetime.now().isoformat()
                save_batch_info(batch_info)
                
                batch_run_details.update({
                    "processing_status": batch_creation_response.processing_status,
                    "created_at_api": to_iso_format_if_datetime(batch_creation_response.created_at),
                    "total_requests_in_batch": batch_creation_response.request_counts.total if hasattr(batch_creation_response.request_counts, 'total') else len(requests_payload)
                })
            
            this_batch_results_dir = os.path.join(results_dir, api_batch_id)
            ensure_dir_exists(this_batch_results_dir)
            details_file_path = os.path.join(this_batch_results_dir, "claude_batch_run_details.json")
            with open(details_file_path, 'w') as f: 
                json.dump(batch_run_details, f, indent=2)

            completed = False
            current_status_obj = None
            while not completed:
                print(f"Checking status of Claude batch {api_batch_id}...")
                current_status_obj = check_claude_batch_status(client, api_batch_id)
                current_processing_status_str = current_status_obj.processing_status
                print(f"Batch {api_batch_id} processing_status: {current_processing_status_str}")
                
                # Update batch state
                batch_info['batch_states'][batch_key]['status'] = current_processing_status_str
                save_batch_info(batch_info)
                
                batch_run_details["processing_status"] = current_processing_status_str
                batch_run_details["updated_at_script"] = to_iso_format_if_datetime(datetime.datetime.now(datetime.timezone.utc))
                rc = current_status_obj.request_counts
                if hasattr(rc, 'model_dump'): 
                    batch_run_details["request_counts"] = rc.model_dump()
                elif rc is not None:
                     try: 
                         batch_run_details["request_counts"] = vars(rc)
                     except TypeError: 
                         batch_run_details["request_counts"] = rc if isinstance(rc, dict) else None
                else: 
                    batch_run_details["request_counts"] = None
                batch_run_details["ended_at_api"] = to_iso_format_if_datetime(getattr(current_status_obj, 'ended_at', None))
                batch_run_details["results_url"] = getattr(current_status_obj, 'results_url', None)
                with open(details_file_path, 'w') as f: 
                    json.dump(batch_run_details, f, indent=2)

                if current_processing_status_str == 'ended':
                    print(f"Batch {api_batch_id} processing ended.")
                    completed = True
                elif current_processing_status_str in ['failed', 'expired', 'canceled']:
                     print(f"Batch {api_batch_id} terminal error state: {current_processing_status_str}")
                     batch_info['batch_states'][batch_key]['status'] = 'failed'
                     batch_info['batch_states'][batch_key]['error_message'] = f"Batch ended with status: {current_processing_status_str}"
                     if batch_run_details["ended_at_api"] is None: 
                         batch_run_details["ended_at_api"] = to_iso_format_if_datetime(datetime.datetime.now(datetime.timezone.utc))
                     if hasattr(current_status_obj, 'failed_at'): 
                         batch_run_details["failed_at_api"] = to_iso_format_if_datetime(current_status_obj.failed_at)
                     with open(details_file_path, 'w') as f: 
                         json.dump(batch_run_details, f, indent=2)
                     save_batch_info(batch_info)
                     raise Exception(f"Claude batch {api_batch_id} ended with status: {current_processing_status_str}")
                else:
                    print(f"Waiting {check_interval}s for batch {api_batch_id}...")
                    time.sleep(check_interval)

            final_batch_status = current_status_obj
            results_text = None
            if final_batch_status.processing_status == 'ended':
                raw_results_file_path = os.path.join(this_batch_results_dir, "raw_results.jsonl")
                try:
                    results_text = stream_and_save_claude_batch_results(client, final_batch_status.id, raw_results_file_path)
                    batch_info['batch_states'][batch_key]['results_file'] = raw_results_file_path
                except Exception as e_stream_dl:
                    print(f"Error SDK result streaming for batch {final_batch_status.id}: {e_stream_dl}")
                    batch_info['batch_states'][batch_key]['status'] = 'failed'
                    batch_info['batch_states'][batch_key]['error_message'] = f"Failed to download results: {e_stream_dl}"
                    save_batch_info(batch_info)
                    return []
                    
                if results_text:
                    print(f"Processing results for batch {final_batch_status.id}...")
                    batch_processed_items = process_downloaded_results(results_text, original_data_map)
                    processed_output_file = os.path.join(this_batch_results_dir, "processed_results_combined.json")
                    with open(processed_output_file, 'w') as f: 
                        json.dump(batch_processed_items, f, indent=2)
                    
                    # Update batch state to completed
                    batch_info['batch_states'][batch_key]['status'] = 'completed'
                    batch_info['batch_states'][batch_key]['completed_at'] = datetime.datetime.now().isoformat()
                    batch_info['batch_states'][batch_key]['processed_file'] = processed_output_file
                    save_batch_info(batch_info)
                    
                    print(f"✓ Saved {len(batch_processed_items)} items for batch {final_batch_status.id} to {processed_output_file}")
                    return batch_processed_items
                else:
                    print(f"Failed to retrieve/process results for batch {final_batch_status.id}.")
                    batch_info['batch_states'][batch_key]['status'] = 'failed'
                    batch_info['batch_states'][batch_key]['error_message'] = 'Failed to retrieve results'
                    save_batch_info(batch_info)
                    return []
            else:
                print(f"Batch {final_batch_status.id} not 'ended'. Status: {final_batch_status.processing_status}. Skipping results.")
                batch_info['batch_states'][batch_key]['status'] = 'failed'
                batch_info['batch_states'][batch_key]['error_message'] = f"Unexpected final status: {final_batch_status.processing_status}"
                save_batch_info(batch_info)
                return []
        except Exception as e:
            print(f"Error processing batch {batch_key}: {type(e).__name__} - {e}")
            error_log_base_dir = os.path.join(results_dir, "script_errors")
            ensure_dir_exists(error_log_base_dir)
            api_batch_id_for_log = batch_run_details.get("api_batch_id") if batch_run_details.get("api_batch_id") else "NO_BATCH_ID"
            error_file_name = f"{api_batch_id_for_log}_{os.path.splitext(batch_file_name)[0]}_error.json"
            error_details_path = os.path.join(error_log_base_dir, error_file_name)
            batch_run_details["processing_status"] = "ERROR_IN_SCRIPT"
            batch_run_details["error_message"] = str(e)
            batch_run_details["error_type"] = type(e).__name__
            batch_run_details["updated_at_script"] = to_iso_format_if_datetime(datetime.datetime.now(datetime.timezone.utc))
            serializable_details = {k: to_iso_format_if_datetime(v) for k, v in batch_run_details.items()}
            try:
                with open(error_details_path, 'w') as ef: 
                    json.dump(serializable_details, ef, indent=2)
                print(f"Error details saved to {error_details_path}")
            except Exception as inner_e: 
                print(f"Additionally, failed to write error details: {inner_e}")
            
            # Update batch state
            batch_info['batch_states'][batch_key]['status'] = 'failed'
            batch_info['batch_states'][batch_key]['error_message'] = str(e)
            save_batch_info(batch_info)
            return []
    
    # Process incomplete batches
    if concurrent_batches <= 1:
        # Process batches sequentially
        for batch_key_and_state in incomplete_batches:
            batch_results = process_single_claude_batch_file(batch_key_and_state)
            all_results.extend(batch_results)
    else:
        # Process batches concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_batches) as executor:
            futures = []
            for batch_key_and_state in incomplete_batches:
                future = executor.submit(process_single_claude_batch_file, batch_key_and_state)
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
    
    print(f"✅ All batch processing done. Processed {len(all_results)} items. Combined results: {output_file}")
    return all_results

def process_response(response):
    """Process Claude API response"""
    return response  # Claude response is already in the right format