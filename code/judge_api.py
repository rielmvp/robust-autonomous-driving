# judge_api.py
import argparse
import json
import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# --- Self-Contained OpenAI API Functions (Simplified for Text-Only) ---
thread_local = threading.local()

def get_client(api_key):
    """Get or create OpenAI client (thread-local)"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = OpenAI(api_key=api_key)
    return thread_local.client

def make_o3_api_call_text_only(api_key, model, prompt, temperature=0.0):
    """Make a text-only OpenAI API call for judging."""
    client = get_client(api_key)
    
    # The content is now just the text prompt
    content = [{"type": "text", "text": prompt}]
    
    # Retry logic for API calls
    for attempt in range(3):
        try:
            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": content}]
            }
            # o3 models do not use the temperature parameter, but we leave the argument
            # in the function signature for consistency, even if not used in the API call itself.
            response = client.chat.completions.create(**api_params)
            return response.model_dump()
            
        except Exception as e:
            print(f"OpenAI API attempt {attempt+1} failed: {e}")
            if attempt < 2:
                wait_time = (2 ** attempt) * 30  # 30s, 60s
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

# --- Common Utility Functions (streamlined for judging task) ---
def read_api_key(key_path):
    """Read API key from file"""
    try:
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key not found at {key_path}.")

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_existing_results(output_file):
    """Load existing results to allow resuming a run"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {output_file} is corrupted. Starting fresh.")
    return []

def get_processed_ids(results):
    """Get a set of IDs from existing results"""
    return {item['ID'] for item in results if 'ID' in item}

def save_results(results, output_file, lock):
    """Thread-safe save results to file"""
    with lock:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

def process_item_for_judging(item, api_key, model, temperature):
    """Process a single item using the text-only OpenAI API for judging"""
    item_id = item['ID']
    try:
        # NO LONGER LOADING IMAGES
        prompt_text = item['prompt'] # This is the pre-formatted judge prompt
        
        # Make the text-only API call to the judging model
        response = make_o3_api_call_text_only(api_key, model, prompt_text, temperature)
        
        # Add the judge's response to the item
        item_with_judgment = item.copy()
        item_with_judgment['judge_response'] = response
        
        print(f"✓ Successfully judged {item_id}")
        return item_with_judgment
        
    except Exception as e:
        print(f"✗ Failed to judge {item_id}: {e}")
        return None

def create_summary_file(output_file, model, successful_items, failed_items, total_results):
    """Create a summary file with judging statistics"""
    summary_file = f"{output_file}.summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Judging Summary\n")
        f.write("================\n")
        f.write(f"Judge Model: {model}\n")
        f.write(f"Total items processed this run: {len(successful_items) + len(failed_items)}\n")
        f.write(f"Successfully judged: {len(successful_items)}\n")
        f.write(f"Failed to judge: {len(failed_items)}\n")
        f.write(f"Total judgments saved: {total_results}\n")
        
        if failed_items:
            f.write("\nFailed item IDs:\n")
            for item_id in failed_items:
                f.write(f"- {item_id}\n")
    print(f"Summary saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Judge model outputs using the OpenAI API (Text-Only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input-file", type=str, required=True, 
                        help="Input JSON file containing items to be judged.")
    
    parser.add_argument("--model", type=str, default="o3", 
                        help="The judging model to use (e.g., 'o3-mini', 'o3').")
                        
    parser.add_argument("--output-file", type=str, default=None, 
                        help="Output JSON file path. If not provided, it will be auto-generated in 'results/judge/'.")
                        
    parser.add_argument("--api-key-path", type=str, default="api_key/openai_key.txt", 
                        help="Path to OpenAI API key file.")
                        
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Temperature for the judging model. Note: o3 models ignore this.")
                        
    parser.add_argument("--concurrency", type=int, default=50, 
                        help="Number of concurrent API calls.")
                        
    parser.add_argument("--test", action="store_true", 
                        help="Test mode with only 5 samples.")
    
    args = parser.parse_args()
    
    # --- Setup ---
    if args.output_file is None:
        output_dir = "results/judge"
        ensure_dir_exists(output_dir)
        base_name = os.path.basename(args.input_file)
        args.output_file = os.path.join(output_dir, f"judge_{base_name}")

    api_key = read_api_key(args.api_key_path)
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)
        
    if args.test:
        data = data[:5]
        print(f"--- TEST MODE: Using first {len(data)} samples ---")

    print(f"Judge Model: {args.model}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print("Mode: Text-Only (no images will be loaded or sent)")

    # --- Processing ---
    results = load_existing_results(args.output_file)
    processed_ids = get_processed_ids(results)
    items_to_process = [item for item in data if item['ID'] not in processed_ids]
    
    print(f"Found {len(results)} existing judgments. {len(items_to_process)} items left to judge.")
    
    if not items_to_process:
        print("All items have already been judged. Exiting.")
        return
        
    results_lock = threading.Lock()
    save_lock = threading.Lock()
    successful_items = []
    failed_items = []

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_item = {
                executor.submit(
                    process_item_for_judging, 
                    item, api_key, args.model, args.temperature
                ): item for item in items_to_process
            }
            
            for i, future in enumerate(as_completed(future_to_item)):
                item_id = future_to_item[future]['ID']
                try:
                    result = future.result()
                    if result:
                        with results_lock:
                            results.append(result)
                            successful_items.append(result['ID'])
                    else:
                        failed_items.append(item_id)
                    
                    if (i + 1) % 10 == 0 or (i + 1) == len(items_to_process):
                        save_results(results, args.output_file, save_lock)
                        total_processed_this_run = len(successful_items) + len(failed_items)
                        print(f"  → Saved progress ({len(results)}/{len(data)} total, {total_processed_this_run}/{len(items_to_process)} this run)")
                        
                except Exception as exc:
                    print(f"Item {item_id} generated an exception: {exc}")
                    failed_items.append(item_id)
    
    finally:
        save_results(results, args.output_file, save_lock)
        create_summary_file(args.output_file, args.model, successful_items, failed_items, len(results))
    
    print("\n--- Judging Complete ---")
    print(f"Successfully judged {len(successful_items)} items.")
    print(f"Failed to judge {len(failed_items)} items.")
    print(f"Final results saved to {args.output_file}")
    if failed_items:
        print("Consider re-running the script to retry failed items.")

if __name__ == "__main__":
    main()