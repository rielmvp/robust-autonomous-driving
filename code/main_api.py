# main_api.py
import argparse
import json
import os
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import importlib

# Provider modules are imported on first use, so only the SDK of the provider being run has to be installed
SINGLE_CALL_MODULES = {'openai': 'openai_api', 'claude': 'claude_api', 'gemini': 'gemini_api'}
BATCH_MODULES = {'openai': 'openai_batch_api', 'claude': 'claude_batch_api'}

# Common utility functions
def get_default_api_key_path(provider):
    """Get default API key path based on provider"""
    provider_key_map = {
        'openai': 'api_key/openai_key.txt',
        'gemini': 'api_key/gemini_key.txt',
        'claude': 'api_key/claude_key.txt'
    }
    return provider_key_map.get(provider.lower(), f'api_key/{provider.lower()}_key.txt')

def read_api_key(key_path, allow_missing=False):
    """Read API key from file"""
    try:
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        if allow_missing:
            # Self-hosted OpenAI-compatible servers (e.g. vLLM) do not need a key
            print(f"No API key at {key_path}; using a placeholder key for the self-hosted server.")
            return "EMPTY"
        raise FileNotFoundError(f"API key not found at {key_path}. Please make sure the file exists.")

def get_model_folder_name(model):
    """Folder-safe model name, e.g. Qwen/Qwen2.5-VL-7B-Instruct -> Qwen2.5-VL-7B-Instruct"""
    return model.rstrip('/').split('/')[-1].replace(':', '_')

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if directory:
        Path(os.path.dirname(directory)).mkdir(parents=True, exist_ok=True)

def load_existing_results(output_file):
    """Load existing results if the file exists"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            return results
        except json.JSONDecodeError:
            print(f"Warning: {output_file} exists but is not valid JSON. Starting fresh.")
    return []

def get_processed_ids(results):
    """Get a set of IDs that have already been processed"""
    return {item['ID'] for item in results if 'ID' in item}

def get_output_filename(task_name, model_name, batch_mode=False):
    """Generate output filename based on task name and model name"""
    mode_suffix = "_batch" if batch_mode else ""
    return f"results/{task_name}/{model_name}/{task_name}_{model_name}{mode_suffix}.json"

def save_results(results, output_file, lock):
    """Thread-safe save results to file"""
    with lock:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

def get_api_module(provider, batch_mode=False):
    """Get the appropriate API module based on provider and mode"""
    modules = BATCH_MODULES if batch_mode else SINGLE_CALL_MODULES
    module_name = modules.get(provider.lower())
    if module_name is None:
        mode_text = "batch mode" if batch_mode else "single-call mode"
        raise ValueError(f"Unsupported provider for {mode_text}: {provider}")
    return importlib.import_module(module_name)

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

def filter_data_by_n_frames(data, n_frames_list):
    """Filter data based on n_frames values"""
    if not n_frames_list or 'all' in n_frames_list:
        return data
    
    # Convert to integers for comparison
    target_frames = []
    for nf in n_frames_list:
        if str(nf).lower() != 'all':
            try:
                target_frames.append(int(nf))
            except ValueError:
                raise ValueError(f"Invalid n_frames value '{nf}'. Must be an integer or 'all'")
    
    # Filter data based on n_frames
    filtered_data = []
    for item in data:
        if 'n_frames' in item and item['n_frames'] in target_frames:
            filtered_data.append(item)
    
    return filtered_data

def parse_n_frames(n_frames_str):
    """Parse comma-separated n_frames string into list"""
    if not n_frames_str:
        return [32]  # Default to 32
    
    # Split by comma and clean up whitespace
    frames = [nf.strip() for nf in n_frames_str.split(',')]
    
    # Validate n_frames values
    validated_frames = []
    for nf in frames:
        if nf.lower() == 'all':
            validated_frames.append('all')
        else:
            try:
                frame_count = int(nf)
                if frame_count <= 0:
                    raise ValueError(f"n_frames must be positive, got {frame_count}")
                validated_frames.append(frame_count)
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid n_frames value '{nf}'. Must be a positive integer or 'all'")
                else:
                    raise e
    
    return validated_frames

def find_existing_batches(task_name, model_name):
    """Find existing batch directories that might contain running batches"""
    batch_base_dir = os.path.join("batches", task_name)
    if not os.path.exists(batch_base_dir):
        return []
    
    existing_batches = []
    for batch_dir in os.listdir(batch_base_dir):
        if model_name in batch_dir:
            batch_path = os.path.join(batch_base_dir, batch_dir)
            batch_info_file = os.path.join(batch_path, "batch_info.json")
            if os.path.exists(batch_info_file):
                try:
                    with open(batch_info_file, 'r') as f:
                        batch_info = json.load(f)
                    batch_info['batch_dir'] = batch_path
                    existing_batches.append(batch_info)
                except json.JSONDecodeError:
                    continue
    
    # Sort by creation time (newest first)
    existing_batches.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return existing_batches

def prompt_for_resume(existing_batches):
    """Prompt user whether to resume existing batches"""
    if not existing_batches:
        return False, None
    
    print(f"\nFound {len(existing_batches)} existing batch run(s):")
    for i, batch_info in enumerate(existing_batches):
        print(f"  {i+1}. {batch_info['batch_dir']} ({batch_info.get('num_batches', 'unknown')} batches)")
        if 'batch_states' in batch_info:
            completed = sum(1 for state in batch_info['batch_states'].values() if state.get('status') == 'completed')
            total = len(batch_info['batch_states'])
            print(f"     Status: {completed}/{total} batches completed")
    
    while True:
        choice = input(f"\nWould you like to resume an existing batch run? (y/n or number 1-{len(existing_batches)}): ").strip().lower()
        
        if choice == 'n' or choice == 'no':
            return False, None
        elif choice == 'y' or choice == 'yes':
            if len(existing_batches) == 1:
                return True, existing_batches[0]
            else:
                print("Please specify which batch to resume (1-{})".format(len(existing_batches)))
                continue
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(existing_batches):
                    return True, existing_batches[idx]
                else:
                    print(f"Please enter a number between 1 and {len(existing_batches)}")
            except ValueError:
                print("Please enter 'y', 'n', or a number")

def process_item(item, api_key, model, provider, temperature, prefix=None, max_tokens=None, openai_base_url=None):
    """Process a single item using the specified API provider (single-call mode)"""
    item_id = item['ID']
    
    try:
        # Get image paths (can be single or multiple images)
        image_paths = get_image_paths_from_item(item)
        
        # Get the appropriate API module
        api_module = get_api_module(provider)
        
        # Prepare prompt
        prompt_text = item['prompt']
        if prefix:
            prompt_text = f"{prefix}\n{prompt_text}"
        
        # Make API call based on provider (prompt_text already includes the prefix)
        if provider.lower() == 'claude':
            response = api_module.make_api_call(
                api_key, model, prompt_text, image_paths, temperature, max_tokens or 2048
            )
        elif provider.lower() == 'openai':
            response = api_module.make_api_call(
                api_key, model, prompt_text, image_paths, temperature,
                max_tokens=max_tokens, base_url=openai_base_url
            )
        else:
            response = api_module.make_api_call(
                api_key, model, prompt_text, image_paths, temperature, max_tokens=max_tokens
            )
        
        # Process response
        processed_response = api_module.process_response(response)
        
        # Create result item
        item_with_response = item.copy()
        if prefix:
            item_with_response['processed_prompt'] = prompt_text
        item_with_response['api_response'] = processed_response
        
        print(f"✓ Successfully processed {item_id}")
        return item_with_response
        
    except Exception as e:
        print(f"✗ Failed to process {item_id}: {e}")
        return None

def process_batch_mode(api_key, data, model, provider, temperature, task_name, output_file, 
                      batch_size_mb=None, num_batches=None, prefix=None, max_tokens=None, 
                      test_mode=False, check_interval=10, concurrent_batches=10, resume_batch_info=None):
    """Process data using batch API with resume support"""
    print(f"Running in BATCH mode for {provider.upper()}")
    
    # Get the batch API module
    batch_api_module = get_api_module(provider, batch_mode=True)
    
    batch_info = resume_batch_info
    
    if resume_batch_info:
        print("Resuming existing batch run...")
        # Use existing batch info
        batch_info = resume_batch_info
        # Verify batch files still exist
        missing_files = []
        for batch_file in batch_info['batch_files']:
            if not os.path.exists(batch_file):
                missing_files.append(batch_file)
        
        if missing_files:
            print(f"Error: Some batch files are missing: {missing_files}")
            print("Cannot resume. Please start a new batch run.")
            return []
    else:
        # Prepare new batch
        print("Preparing new batch files...")
        if provider.lower() == 'claude':
            batch_info = batch_api_module.prepare_batch(
                data, model, temperature, task_name, max_tokens or 2048, batch_size_mb, num_batches, prefix, test_mode
            )
        else:  # OpenAI
            batch_info = batch_api_module.prepare_batch(
                data, model, temperature, task_name, batch_size_mb, num_batches, prefix, test_mode, max_tokens
            )
    
    # Add input file info to batch_info for processing
    input_file = os.path.join(batch_info['batch_dir'], 'temp_input.json')
    batch_info['input_file'] = input_file
    with open(input_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Submit and process batches with resume support
    print("Submitting batches and waiting for results...")
    results = batch_api_module.submit_and_process_batches(
        api_key, batch_info, output_file, check_interval, concurrent_batches
    )
    
    # Clean up temporary file
    if os.path.exists(input_file):
        os.remove(input_file)
    
    return results

def create_summary_file(output_file, model, provider, prefix, temperature, successful_items, failed_items, total_results, batch_mode=False, n_frames=None):
    """Create a summary file with processing statistics"""
    mode_text = "BATCH" if batch_mode else "SINGLE-CALL"
    summary_file = f"{output_file}.summary.txt"
    try:
        with open(summary_file, 'w') as f:
            f.write(f"Processing Summary ({mode_text} Mode)\n")
            f.write(f"================================\n")
            f.write(f"Provider: {provider}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Mode: {mode_text}\n")
            if prefix:
                f.write(f"Prefix: \"{prefix}\"\n")
            if n_frames and 'all' not in n_frames:
                f.write(f"N-Frames: {', '.join(map(str, n_frames))}\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"Total items processed this run: {len(successful_items) + len(failed_items)}\n")
            f.write(f"Successfully processed: {len(successful_items)}\n")
            f.write(f"Failed to process: {len(failed_items)}\n")
            f.write(f"Total results saved: {total_results}\n")
            
            if failed_items:
                f.write("\nFailed items:\n")
                for item in failed_items:
                    f.write(f"- {item}\n")
        print(f"Summary saved to {summary_file}")
    except Exception as e:
        print(f"Failed to write summary file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Make API calls for image analysis with multiple providers")
    
    # Provider selection
    parser.add_argument("--provider", type=str, choices=['openai', 'claude', 'gemini'], default='openai', 
                       help="API provider to use")
    
    # Common arguments
    parser.add_argument("--model", type=str, default="gpt-4.1", help="Model to use")
    parser.add_argument("--openai-base-url", type=str, default=None,
                       help="Base URL of an OpenAI-compatible server (e.g. vLLM at http://localhost:8000/v1)")
    parser.add_argument("--input-file", type=str, default="dataset/radbench.json", help="Input JSON file path")
    parser.add_argument("--output-file", type=str, default=None, 
                       help="Output JSON file path (optional, will be auto-generated if not provided)")
    parser.add_argument("--api-key-path", type=str, default=None, 
                       help="Path to API key file (optional, will be auto-generated based on provider if not provided)")
    parser.add_argument("--task-name", type=str, default="16_frames", help="Task name for organizing results")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument("--test", action="store_true", help="Test mode with only 30 samples")
    parser.add_argument("--concurrency", type=int, default=300, help="Number of concurrent API calls (single-call mode)")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix to add to all prompts")
    parser.add_argument("--n-frames", type=str, default="16", 
                       help="Number of frames per scenario: 16 (main benchmark), 4, 8 or 32, a comma-separated list, or 'all'")
    parser.add_argument("--max-tokens", type=int, default=None, 
                       help="Max tokens for the model response (default: provider default; 2048 for Claude, which requires a limit)")
    
    # Batch mode arguments
    parser.add_argument("--batch", action="store_true", help="Use batch API instead of single calls")
    parser.add_argument("--batch-size-mb", type=float, default=185, 
                       help="Maximum batch size in MB (batch mode only)")
    parser.add_argument("--num-batches", type=int, default=None, 
                       help="Number of batches to create (batch mode only)")
    parser.add_argument("--check-interval", type=int, default=10, 
                       help="Interval in seconds to check batch status (batch mode only)")
    parser.add_argument("--concurrent-batches", type=int, default=50, 
                       help="Number of batches to process concurrently (batch mode only)")
    
    # Resume arguments
    parser.add_argument("--no-resume", action="store_true", 
                       help="Skip checking for existing batches and start fresh")
    
    args = parser.parse_args()
    
    # Parse and validate n_frames
    try:
        n_frames_list = parse_n_frames(args.n_frames)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Set API key path based on provider if not explicitly provided
    if args.api_key_path is None:
        args.api_key_path = get_default_api_key_path(args.provider)
    
    # Fail early on an unsupported provider/mode combination
    get_api_module(args.provider, batch_mode=args.batch)
    model_folder_name = get_model_folder_name(args.model)
    
    # Generate output filename
    output_file = args.output_file if args.output_file else get_output_filename(args.task_name, model_folder_name, args.batch)
    ensure_dir_exists(output_file)
    
    # Read API key
    api_key = read_api_key(args.api_key_path, allow_missing=bool(args.openai_base_url))

    # Load input data
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Filter data by n_frames
    original_count = len(data)
    data = filter_data_by_n_frames(data, n_frames_list)
    if 'all' not in n_frames_list:
        print(f"Filtered data: {original_count} -> {len(data)} items (n_frames: {', '.join(map(str, n_frames_list))})")

    if args.test:
        data = data[:30]  # Changed from 2 to 30
        print(f"Test mode: Using first {len(data)} samples")
    
    mode_text = "BATCH" if args.batch else "SINGLE-CALL"
    print(f"Provider: {args.provider.upper()}")
    print(f"Model: {args.model}")
    print(f"Mode: {mode_text}")
    print(f"Task: {args.task_name}")
    print(f"API Key Path: {args.api_key_path}")
    print(f"Using output file: {output_file}")
    if args.prefix:
        print(f"Using prefix: \"{args.prefix}\"")
    if 'all' not in n_frames_list:
        print(f"N-frames: {', '.join(map(str, n_frames_list))}")
    
    successful_items = []
    failed_items = []
    resume_batch_info = None
    
    if args.batch:
        # Batch processing mode
        # Check for existing batches (unless --no-resume is specified)
        if not args.no_resume:
            existing_batches = find_existing_batches(args.task_name, model_folder_name)
            if existing_batches:
                should_resume, resume_batch_info = prompt_for_resume(existing_batches)
                if not should_resume:
                    resume_batch_info = None
        
        # Set default batch parameters based on provider
        if args.batch_size_mb is None:
            args.batch_size_mb = 185 if args.provider.lower() == 'openai' else 240  # Claude
        
        if resume_batch_info:
            print(f"Resuming batch processing from: {resume_batch_info['batch_dir']}")
        else:
            print(f"Starting new batch processing with max batch size: {args.batch_size_mb}MB")
            if args.num_batches:
                print(f"Target number of batches: {args.num_batches}")
        
        try:
            results = process_batch_mode(
                api_key, data, args.model, args.provider, args.temperature, 
                args.task_name, output_file, args.batch_size_mb, args.num_batches, 
                args.prefix, args.max_tokens, args.test, args.check_interval, 
                args.concurrent_batches, resume_batch_info
            )
            successful_items = [item['ID'] for item in results if 'ID' in item]
        except Exception as e:
            print(f"Batch processing failed: {e}")
            failed_items = [item['ID'] for item in data]
            results = []
    else:
        # Single-call processing mode
        # Load existing results and determine what needs processing
        results = load_existing_results(output_file)
        processed_ids = get_processed_ids(results)
        items_to_process = [item for item in data if item['ID'] not in processed_ids]
        
        print(f"Found {len(results)} existing results. {len(items_to_process)} items left to process.")
        print(f"Running with concurrency: {args.concurrency}")
        
        if not items_to_process:
            print("All items have already been processed. Nothing to do.")
            return
        
        # Initialize threading locks and counters
        results_lock = threading.Lock()
        save_lock = threading.Lock()
        
        try:
            # Process items with thread pool
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(
                        process_item, 
                        item, 
                        api_key, 
                        args.model, 
                        args.provider,
                        args.temperature,
                        args.prefix,
                        args.max_tokens,
                        args.openai_base_url
                    ): item for item in items_to_process
                }
                
                # Process completed tasks
                for i, future in enumerate(as_completed(future_to_item)):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        if result:
                            with results_lock:
                                results.append(result)
                                successful_items.append(result['ID'])
                        else:
                            failed_items.append(item['ID'])
                        
                        # Save progress periodically
                        if (i + 1) % 5 == 0 or i == len(items_to_process) - 1:
                            save_results(results, output_file, save_lock)
                            processed_count = len(successful_items) + len(failed_items)
                            print(f"  → Saved progress ({len(results)}/{len(data)} items, {processed_count}/{len(items_to_process)} this run)")
                            
                    except Exception as exc:
                        print(f"Item {item['ID']} generated an exception: {exc}")
                        failed_items.append(item['ID'])
        
        except Exception as e:
            print(f"Exception in main processing loop: {e}")
        
        finally:
            # Final save
            save_results(results, output_file, save_lock)
    
    # Create summary
    create_summary_file(output_file, args.model, args.provider, args.prefix, 
                       args.temperature, successful_items, failed_items, len(results if 'results' in locals() else []), args.batch, n_frames_list)
    
    print(f"All done! Successfully processed {len(successful_items)} items, failed {len(failed_items)}.")
    print(f"Results saved to {output_file}")
    
    if failed_items:
        print(f"Some items failed to process. Consider running the script again to retry these items.")

if __name__ == "__main__":
    main()